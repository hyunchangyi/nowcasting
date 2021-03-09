function S = FIS(A, S)
%FIS()    Applies fixed-interval smoother
%
%  Syntax:
%    S = FIS(A, S)
%
%  Description:
%    SKF() applies a fixed-interval smoother, and is used in conjunction 
%    with SKF(). Starting from the final value, this uses a set of 
%    recursions and works backwards. Smoothers are advantageous in that 
%    future values are used for estimation. See  page 154 of 'Forecasting, 
%    structural time series models and the Kalman filter' for more details 
%    (Harvey, 1990).
%
%  Input parameters:
%    A: Transition matrix coefficients. 
%    S: Output from SKF() step. See SKF() for details.
%
%  Output parameters:
%    S: In addition to the output from SKF(), FIS() adds the following
%       smoothed estimates. Note that t = 1...T gives time:
%    - S.ZmT: Smoothed estimate of factor values (Z_t|T) 
%    - S.VmT: Smoothed estimate of factor covariance matrix (V_t|T = Cov(Z_t|T))
%    - S.VmT_1: Smoothed estimate of Lag 1 factor covariance matrix (Cov(Z_tZ_t-1|T))
%
%  Model:
%   Y_t = C_t Z_t + e_t for e_t ~ N(0, R)
%   Z_t = A Z_{t-1} + mu_t for mu_t ~ N(0, Q)

%% ORGANIZE INPUT ---------------------------------------------------------

% Initialize output matrices
  [m, nobs] = size(S.Zm);
  S.ZmT = zeros(m,nobs+1);
  S.VmT = zeros(m,m,nobs+1);
  
  % Fill the final period of ZmT, VmT with SKF() posterior values
  S.ZmT(:,nobs+1)   = squeeze(S.ZmU(:, nobs+1));
  S.VmT(:,:,nobs+1) = squeeze(S.VmU(:,:, nobs+1));

  % Initialize VmT_1 lag 1 covariance matrix for final period
  S.VmT_1(:,:,nobs) = (eye(m)-S.k_t) *A*squeeze(S.VmU(:,:,nobs));
  
  % Used for recursion process. See companion file for details
  J_2 = squeeze(S.VmU(:,:,nobs)) * A' * pinv(squeeze(S.Vm(:,:,nobs)));

  %% RUN SMOOTHING ALGORITHM ----------------------------------------------
  
  % Loop through time reverse-chronologically (starting at final period nobs)
    for t = nobs:-1:1
                
        % Store posterior and prior factor covariance values 
        VmU = squeeze(S.VmU(:,:,t));
        Vm1 = squeeze(S.Vm(:,:,t));
        
        % Store previous period smoothed factor covariance and lag-1 covariance
        V_T = squeeze(S.VmT(:,:,t+1));
        V_T1 = squeeze(S.VmT_1(:,:,t));
      
        J_1 = J_2;
                
        % Update smoothed factor estimate
        S.ZmT(:,t) = S.ZmU(:,t) + J_1 * (S.ZmT(:,t+1) - A * S.ZmU(:,t)) ; 
        
        % Update smoothed factor covariance matrix
        S.VmT(:,:,t) = VmU + J_1 * (V_T - Vm1) * J_1';   
      
        if t>1
            % Update weight
            J_2 = squeeze(S.VmU(:, :, t-1)) * A' * pinv(squeeze(S.Vm(:,:,t-1)));
            
            % Update lag 1 factor covariance matrix 
            S.VmT_1(:,:,t-1) = VmU * J_2'+J_1 * (V_T1 - A * VmU) * J_2';
        end
    end
    
end

