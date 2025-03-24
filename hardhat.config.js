require("@nomicfoundation/hardhat-toolbox");
require('dotenv').config();

// Use a default private key if none is provided
const PRIVATE_KEY = process.env.PRIVATE_KEY || "0x0000000000000000000000000000000000000000000000000000000000000001";
const POLYGON_MUMBAI_RPC_URL = process.env.POLYGON_MUMBAI_RPC_URL || "https://rpc-mumbai.maticvigil.com";
const ETHEREUM_GOERLI_RPC_URL = process.env.ETHEREUM_GOERLI_RPC_URL || "https://goerli.infura.io/v3/your-api-key";

/** @type import('hardhat/config').HardhatUserConfig */
module.exports = {
  solidity: {
    version: "0.8.28",
    settings: {
      optimizer: {
        enabled: true,
        runs: 200
      }
    }
  },
  networks: {
    hardhat: {
      chainId: 31337
    },
    localhost: {
      chainId: 31337
    },
    // Only use these networks when you have the proper configuration
    ...(PRIVATE_KEY.length >= 64 ? {
      mumbai: {
        url: POLYGON_MUMBAI_RPC_URL,
        accounts: [PRIVATE_KEY],
        chainId: 80001,
        gasPrice: 35000000000
      },
      goerli: {
        url: ETHEREUM_GOERLI_RPC_URL,
        accounts: [PRIVATE_KEY],
        chainId: 5
      }
    } : {})
  },
  // Rest of your config...
};
