// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";
import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/utils/Address.sol";

/**
 * @title FlashLoanSafetyNet
 * @dev A contract that provides flash loan simulation and risk assessment
 * for multiple DeFi protocols and assets
 */
interface IFlashLoanReceiver {
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external returns (bool);
}

interface IPriceOracle {
    function getAssetPrice(address asset) external view returns (uint256);
    function getAssetsPrices(address[] calldata assets) external view returns (uint256[] memory);
}

contract FlashLoanSafetyNet is Ownable, Pausable {
    using Address for address;

    // Constants
    uint256 public constant FLASH_LOAN_FEE = 9; // 0.09% fee
    uint256 public constant FEE_PRECISION = 10000; // Fee precision

    // State variables
    mapping(address => bool) public guardians;
    mapping(address => bool) public supportedTokens;
    mapping(address => uint256) public tokenLiquidity;
    mapping(address => uint256) public riskScores; // 0-100, higher means riskier

    IPriceOracle public priceOracle;

    // Events
    event LoanChecked(address indexed borrower, address[] tokens, uint256[] amounts, bool safe, uint256 riskScore);
    event LoanSimulated(address indexed borrower, address[] tokens, uint256[] amounts, bool success);
    event GuardianAdded(address indexed guardian);
    event GuardianRemoved(address indexed guardian);
    event EmergencyTriggered(address indexed guardian, string reason);
    event TokenAdded(address indexed token);
    event TokenRemoved(address indexed token);
    event LiquidityUpdated(address indexed token, uint256 amount);
    event RiskScoreUpdated(address indexed token, uint256 score);
    event OracleUpdated(address indexed oracle);

    // Modifiers
    modifier onlyGuardian() {
        require(guardians[msg.sender], "Not an authorized guardian");
        _;
    }

    modifier tokenSupported(address token) {
        require(supportedTokens[token], "Token not supported");
        _;
    }

    constructor() {
        guardians[msg.sender] = true;
    }

    /**
     * @dev Add a guardian who can trigger emergency actions
     * @param _guardian Address of the guardian to add
     */
    function addGuardian(address _guardian) external onlyOwner {
        require(_guardian != address(0), "Invalid guardian address");
        guardians[_guardian] = true;
        emit GuardianAdded(_guardian);
    }

    /**
     * @dev Remove a guardian
     * @param _guardian Address of the guardian to remove
     */
    function removeGuardian(address _guardian) external onlyOwner {
        guardians[_guardian] = false;
        emit GuardianRemoved(_guardian);
    }

    /**
     * @dev Set the price oracle contract
     * @param _oracle Address of the price oracle
     */
    function setPriceOracle(address _oracle) external onlyOwner {
        require(_oracle != address(0), "Invalid oracle address");
        require(_oracle.isContract(), "Oracle must be a contract");
        priceOracle = IPriceOracle(_oracle);
        emit OracleUpdated(_oracle);
    }

    /**
     * @dev Add a supported token
     * @param _token Address of the token to add
     * @param _initialLiquidity Initial liquidity amount
     * @param _riskScore Risk score for the token (0-100)
     */
    function addSupportedToken(
        address _token,
        uint256 _initialLiquidity,
        uint256 _riskScore
    ) external onlyOwner {
        require(_token != address(0), "Invalid token address");
        require(_token.isContract(), "Token must be a contract");
        require(_riskScore <= 100, "Risk score must be between 0-100");

        supportedTokens[_token] = true;
        tokenLiquidity[_token] = _initialLiquidity;
        riskScores[_token] = _riskScore;

        emit TokenAdded(_token);
        emit LiquidityUpdated(_token, _initialLiquidity);
        emit RiskScoreUpdated(_token, _riskScore);
    }

    /**
     * @dev Remove a supported token
     * @param _token Address of the token to remove
     */
    function removeSupportedToken(address _token) external onlyOwner {
        supportedTokens[_token] = false;
        emit TokenRemoved(_token);
    }

    /**
     * @dev Update liquidity for a token
     * @param _token Address of the token
     * @param _liquidity New liquidity amount
     */
    function updateLiquidity(address _token, uint256 _liquidity)
        external
        onlyOwner
        tokenSupported(_token)
    {
        tokenLiquidity[_token] = _liquidity;
        emit LiquidityUpdated(_token, _liquidity);
    }

    /**
     * @dev Update risk score for a token
     * @param _token Address of the token
     * @param _riskScore New risk score (0-100)
     */
    function updateRiskScore(address _token, uint256 _riskScore)
        external
        onlyOwner
        tokenSupported(_token)
    {
        require(_riskScore <= 100, "Risk score must be between 0-100");
        riskScores[_token] = _riskScore;
        emit RiskScoreUpdated(_token, _riskScore);
    }

    /**
     * @dev Check if a flash loan is safe based on multiple factors
     * @param borrower Address of the borrower
     * @param tokens Array of token addresses
     * @param amounts Array of token amounts
     * @return safe Whether the loan is considered safe
     * @return riskScore The calculated risk score
     */
    function checkFlashLoan(
        address borrower,
        address[] calldata tokens,
        uint256[] calldata amounts
    )
        external
        whenNotPaused
        returns (bool safe, uint256 riskScore)
    {
        require(tokens.length == amounts.length, "Arrays length mismatch");
        require(tokens.length > 0, "Empty arrays");

        (safe, riskScore) = _assessLoanRisk(borrower, tokens, amounts);
        emit LoanChecked(borrower, tokens, amounts, safe, riskScore);
        return (safe, riskScore);
    }

    /**
     * @dev Simulate a flash loan execution
     * @param receiver Address of the flash loan receiver contract
     * @param tokens Array of token addresses
     * @param amounts Array of token amounts
     * @param params Additional parameters to pass to the receiver
     * @return success Whether the simulation was successful
     */
    function simulateFlashLoan(
        address receiver,
        address[] calldata tokens,
        uint256[] calldata amounts,
        bytes calldata params
    )
        external
        whenNotPaused
        returns (bool success)
    {
        require(tokens.length == amounts.length, "Arrays length mismatch");
        require(tokens.length > 0, "Empty arrays");
        require(receiver.isContract(), "Receiver must be a contract");

        // Check if all tokens are supported
        for (uint256 i = 0; i < tokens.length; i++) {
            require(supportedTokens[tokens[i]], "Unsupported token");
            require(amounts[i] <= tokenLiquidity[tokens[i]], "Insufficient liquidity");
        }

        // Calculate premiums
        uint256[] memory premiums = new uint256[](tokens.length);
        for (uint256 i = 0; i < tokens.length; i++) {
            premiums[i] = (amounts[i] * FLASH_LOAN_FEE) / FEE_PRECISION;
        }

        // Execute the flash loan simulation
        try IFlashLoanReceiver(receiver).executeOperation(
            tokens,
            amounts,
            premiums,
            msg.sender,
            params
        ) returns (bool result) {
            success = result;
        } catch {
            success = false;
        }

        emit LoanSimulated(msg.sender, tokens, amounts, success);
        return success;
    }

    /**
     * @dev Trigger emergency pause with a reason
     * @param reason Reason for triggering the emergency
     */
    function triggerEmergency(string calldata reason) external onlyGuardian {
        _pause();
        emit EmergencyTriggered(msg.sender, reason);
    }

    /**
     * @dev Resolve emergency and unpause the contract
     */
    function resolveEmergency() external onlyOwner {
        _unpause();
    }

    /**
     * @dev Internal function to assess loan risk
     * @param borrower Address of the borrower
     * @param tokens Array of token addresses
     * @param amounts Array of token amounts
     * @return safe Whether the loan is considered safe
     * @return riskScore The calculated risk score
     */
    function _assessLoanRisk(
        address borrower,
        address[] calldata tokens,
        uint256[] calldata amounts
    )
        internal
        view
        returns (bool safe, uint256 riskScore)
    {
        uint256 totalRiskScore = 0;
        uint256 totalValue = 0;

        // Get token prices if oracle is set
        uint256[] memory prices;
        if (address(priceOracle) != address(0)) {
            prices = priceOracle.getAssetsPrices(tokens);
        }

        for (uint256 i = 0; i < tokens.length; i++) {
            // Skip unsupported tokens
            if (!supportedTokens[tokens[i]]) continue;

            // Calculate token risk contribution
            uint256 tokenRisk = riskScores[tokens[i]];

            // Check if amount exceeds available liquidity
            if (amounts[i] > tokenLiquidity[tokens[i]]) {
                tokenRisk = 100; // Maximum risk
            }

            // Calculate token value if oracle is available
            uint256 tokenValue = 0;
            if (address(priceOracle) != address(0) && i < prices.length) {
                tokenValue = amounts[i] * prices[i];
                totalValue += tokenValue;
            } else {
                tokenValue = amounts[i];
                totalValue += tokenValue;
            }

            // Weight risk by token value
            totalRiskScore += tokenRisk * tokenValue;
        }

        // Calculate weighted average risk score
        if (totalValue > 0) {
            riskScore = totalRiskScore / totalValue;
        } else {
            riskScore = 0;
        }

        // Determine if loan is safe (risk score below 70)
        safe = riskScore < 70;

        return (safe, riskScore);
    }
}
