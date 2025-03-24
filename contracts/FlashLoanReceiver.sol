// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";

/**
 * @title FlashLoanReceiver
 * @dev A sample contract that receives and processes flash loans
 * This can be used for testing and simulation purposes
 */
contract FlashLoanReceiver {
    using SafeERC20 for IERC20;
    
    address public owner;
    
    // Events
    event FlashLoanReceived(address[] tokens, uint256[] amounts, uint256[] premiums);
    event FlashLoanCompleted(bool success);
    
    constructor() {
        owner = msg.sender;
    }
    
    /**
     * @dev This function is called after your contract has received the flash loaned amount
     * @param assets The addresses of the assets being flash-borrowed
     * @param amounts The amounts of the assets being flash-borrowed
     * @param premiums The premiums/fees for each asset
     * @param initiator The address initiating the flash loan
     * @param params Arbitrary bytes-encoded params passed from the initiator
     * @return success Whether the operation was successful
     */
    function executeOperation(
        address[] calldata assets,
        uint256[] calldata amounts,
        uint256[] calldata premiums,
        address initiator,
        bytes calldata params
    ) external returns (bool) {
        // Emit event for received flash loan
        emit FlashLoanReceived(assets, amounts, premiums);
        
        // Parse params if needed
        // This is where custom logic would go in a real implementation
        
        // For simulation purposes, we'll just return success
        // In a real implementation, you would perform arbitrage, liquidations, etc.
        
        // Approve the SafetyNet contract to pull the owed amount (amount + premium)
        for (uint256 i = 0; i < assets.length; i++) {
            uint256 amountOwed = amounts[i] + premiums[i];
            IERC20(assets[i]).safeApprove(msg.sender, amountOwed);
        }
        
        emit FlashLoanCompleted(true);
        return true;
    }
    
    /**
     * @dev Allows the owner to rescue tokens sent to this contract
     * @param token The token to rescue
     * @param to The address to send the tokens to
     * @param amount The amount of tokens to rescue
     */
    function rescueTokens(address token, address to, uint256 amount) external {
        require(msg.sender == owner, "Only owner");
        IERC20(token).safeTransfer(to, amount);
    }
}