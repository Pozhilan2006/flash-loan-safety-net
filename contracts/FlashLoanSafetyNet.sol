// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

contract FlashLoanSafetyNet is Ownable, Pausable {
    mapping(address => bool) public guardians;

    event LoanChecked(address indexed borrower, uint256 amount, bool safe);
    event GuardianAdded(address indexed guardian);
    event GuardianRemoved(address indexed guardian);
    event EmergencyTriggered(address indexed guardian);

    modifier onlyGuardian() {
        require(guardians[msg.sender], "Not an authorized guardian");
        _;
    }

    constructor() {
        guardians[msg.sender] = true;
    }

    function addGuardian(address _guardian) external onlyOwner {
        guardians[_guardian] = true;
        emit GuardianAdded(_guardian);
    }

    function removeGuardian(address _guardian) external onlyOwner {
        guardians[_guardian] = false;
        emit GuardianRemoved(_guardian);
    }

    function checkFlashLoan(address borrower, uint256 amount) external whenNotPaused returns (bool) {
        bool safe = _isLoanSafe(borrower, amount);
        emit LoanChecked(borrower, amount, safe);
        return safe;
    }

    function triggerEmergency() external onlyGuardian {
        _pause();
        emit EmergencyTriggered(msg.sender);
    }

    function resolveEmergency() external onlyOwner {
        _unpause();
    }

    function _isLoanSafe(address borrower, uint256 amount) internal pure returns (bool) {
        return amount < 100 ether; // Example condition
    }
}
