const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying contract with the account:", deployer.address);

  // Get the contract factory
  const FlashLoanSafetyNet = await hre.ethers.getContractFactory("FlashLoanSafetyNet");

  // Deploy the contract
  const flashLoanSafetyNet = await FlashLoanSafetyNet.deploy();

  // Wait for deployment confirmation
  await flashLoanSafetyNet.waitForDeployment();

  // Get the contract address
  console.log("FlashLoanSafetyNet deployed to:", await flashLoanSafetyNet.getAddress());
}

main().catch((error) => {
  console.error("Deployment failed:", error);
  process.exitCode = 1;
});
