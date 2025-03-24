// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title FlashLoanPriceOracle
 * @dev A contract that provides price data for assets used in flash loans
 * This oracle can be updated by authorized data providers
 */
contract FlashLoanPriceOracle is Ownable {
    // Mapping from asset address to its price in USD (with 8 decimals)
    mapping(address => uint256) private assetPrices;
    
    // Mapping of authorized data providers
    mapping(address => bool) public dataProviders;
    
    // Events
    event PriceUpdated(address indexed asset, uint256 price);
    event DataProviderAdded(address indexed provider);
    event DataProviderRemoved(address indexed provider);
    
    // Modifiers
    modifier onlyDataProvider() {
        require(dataProviders[msg.sender] || msg.sender == owner(), "Not authorized");
        _;
    }
    
    constructor() {
        // Add contract deployer as a data provider
        dataProviders[msg.sender] = true;
    }
    
    /**
     * @dev Add a data provider
     * @param provider Address of the provider to add
     */
    function addDataProvider(address provider) external onlyOwner {
        require(provider != address(0), "Invalid provider address");
        dataProviders[provider] = true;
        emit DataProviderAdded(provider);
    }
    
    /**
     * @dev Remove a data provider
     * @param provider Address of the provider to remove
     */
    function removeDataProvider(address provider) external onlyOwner {
        dataProviders[provider] = false;
        emit DataProviderRemoved(provider);
    }
    
    /**
     * @dev Update the price of an asset
     * @param asset Address of the asset
     * @param price New price (with 8 decimals)
     */
    function updateAssetPrice(address asset, uint256 price) external onlyDataProvider {
        require(asset != address(0), "Invalid asset address");
        assetPrices[asset] = price;
        emit PriceUpdated(asset, price);
    }
    
    /**
     * @dev Update prices for multiple assets at once
     * @param assets Array of asset addresses
     * @param prices Array of prices (with 8 decimals)
     */
    function updateAssetPrices(address[] calldata assets, uint256[] calldata prices) external onlyDataProvider {
        require(assets.length == prices.length, "Arrays length mismatch");
        
        for (uint256 i = 0; i < assets.length; i++) {
            require(assets[i] != address(0), "Invalid asset address");
            assetPrices[assets[i]] = prices[i];
            emit PriceUpdated(assets[i], prices[i]);
        }
    }
    
    /**
     * @dev Get the price of an asset
     * @param asset Address of the asset
     * @return price Price of the asset (with 8 decimals)
     */
    function getAssetPrice(address asset) external view returns (uint256) {
        return assetPrices[asset];
    }
    
    /**
     * @dev Get prices for multiple assets at once
     * @param assets Array of asset addresses
     * @return prices Array of prices (with 8 decimals)
     */
    function getAssetsPrices(address[] calldata assets) external view returns (uint256[] memory) {
        uint256[] memory prices = new uint256[](assets.length);
        
        for (uint256 i = 0; i < assets.length; i++) {
            prices[i] = assetPrices[assets[i]];
        }
        
        return prices;
    }
}