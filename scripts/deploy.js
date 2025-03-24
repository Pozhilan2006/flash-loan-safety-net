const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying contracts with the account:", deployer.address);

  // Deploy the Price Oracle first
  console.log("Deploying FlashLoanPriceOracle...");
  const FlashLoanPriceOracle = await hre.ethers.getContractFactory("FlashLoanPriceOracle");
  const priceOracle = await FlashLoanPriceOracle.deploy();
  await priceOracle.waitForDeployment();

  const priceOracleAddress = await priceOracle.getAddress();
  console.log(`FlashLoanPriceOracle deployed to: ${priceOracleAddress}`);

  // Deploy the Flash Loan Safety Net
  console.log("Deploying FlashLoanSafetyNet...");
  const FlashLoanSafetyNet = await hre.ethers.getContractFactory("FlashLoanSafetyNet");
  const safetyNet = await FlashLoanSafetyNet.deploy();
  await safetyNet.waitForDeployment();

  const safetyNetAddress = await safetyNet.getAddress();
  console.log(`FlashLoanSafetyNet deployed to: ${safetyNetAddress}`);

  // Set the price oracle in the safety net
  console.log("Setting price oracle in FlashLoanSafetyNet...");
  const setPriceOracleTx = await safetyNet.setPriceOracle(priceOracleAddress);
  await setPriceOracleTx.wait();
  console.log("Price oracle set successfully");

  // Deploy the Flash Loan Receiver (for testing)
  console.log("Deploying FlashLoanReceiver...");
  const FlashLoanReceiver = await hre.ethers.getContractFactory("FlashLoanReceiver");
  const receiver = await FlashLoanReceiver.deploy();
  await receiver.waitForDeployment();

  const receiverAddress = await receiver.getAddress();
  console.log(`FlashLoanReceiver deployed to: ${receiverAddress}`);

  // Log all deployed contract addresses
  console.log("\nDeployment Summary:");
  console.log("-------------------");
  console.log(`FlashLoanPriceOracle: ${priceOracleAddress}`);
  console.log(`FlashLoanSafetyNet: ${safetyNetAddress}`);
  console.log(`FlashLoanReceiver: ${receiverAddress}`);

  // Save deployment addresses to a file for easy reference
  const fs = require("fs");
  const deploymentInfo = {
    network: hre.network.name,
    chainId: hre.network.config.chainId,
    contracts: {
      FlashLoanPriceOracle: priceOracleAddress,
      FlashLoanSafetyNet: safetyNetAddress,
      FlashLoanReceiver: receiverAddress
    },
    timestamp: new Date().toISOString()
  };

  fs.writeFileSync(
    "deployment-info.json",
    JSON.stringify(deploymentInfo, null, 2)
  );

  console.log("\nDeployment information saved to deployment-info.json");
}

main().catch((error) => {
  console.error("Deployment failed:", error);
  process.exitCode = 1;
});
