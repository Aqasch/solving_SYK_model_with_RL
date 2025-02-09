import numpy as np
from scipy.linalg import expm, logm

def helmholtz_free_energy(H, beta):
    """
    Calculate the Helmholtz free energy of a Hamiltonian.
    
    Args:
        H (numpy.ndarray): The Hamiltonian matrix (Hermitian).
        temperature (float): Temperature (in Kelvin).
        kb (float): Boltzmann constant (default in J/K).
    
    Returns:
        float: Helmholtz free energy.
    """
    
    # Compute the thermal state: rho = exp(-beta * H) / Z
    exp_beta_H = expm(-beta * H)
    Z = np.trace(exp_beta_H)
    rho = exp_beta_H / Z  # Density matrix of the thermal state
    
    # Compute expectation value <H> = Tr(rho * H)
    expected_H = np.trace(rho @ H)
    
    # Compute entropy S = -Tr(rho * log(rho))
    # Use logm for matrix logarithm
    rho_log_rho = rho @ logm(rho)
    entropy = -np.trace(rho_log_rho)
    
    # Calculate the Helmholtz free energy: F = <H> - T * S
    F = expected_H - (1/beta)* entropy
    return F.real, expected_H.real, entropy.real, rho
