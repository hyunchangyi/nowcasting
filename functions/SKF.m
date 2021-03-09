function S = SKF(Y, A, C, Q, R, Z_0, V_0)
%SKF    Applies Kalman filter
%
%  Syntax:
%    S = SKF(Y, A, C, Q, R, Z_0, V_0)
%
%  Description:
%    SKF() applies a Kalman filter. This is a bayesian algorithm. Looping 
%    forward in time, a 'prior' estimate is calculated from the previous 
%    period. Then, using the observed data, a 'posterior' value is obtained.
%
%  Input parameters:
%    y:   Input data.
%    A:   Transition matrix coefficients. 
%    C:   Observation matrix coefficients.
%    Q:   Covariance matrix (factors and idiosyncratic component)
%    R:   Variance-Covariance for observation equation residuals 
%    Z_0: Initial factor values
%    V_0: Initial factor covariance matrix 
%
%  Output parameters:
%    S.Zm:     Prior/predicted factor state vector (Z_t|t-1)  
%    S.ZmU:    Posterior/updated state vector (Z_t|t)  
%    S.Vm:     Prior/predicted covariance of factor state vector (V_t|t-1)  
%    S.VmU:    Posterior/updated covariance of factor state vector (V_t|t)  
%    S.loglik: Value of likelihood function
%    S.k_t:    Kalman gain
%
%  Model:
%   Y_t = C_t Z_t + e_t,     e_t ~ N(0, R)
%   Z_t = A Z_{t-1} + mu_t,  mu_t ~ N(0, Q)
  
%% INSTANTIATE OUTPUT VALUES ---------------------------------------------
  % Output structure & dimensions of state space matrix
  [~, m] = size(C);
  
  % Outputs time for data matrix. "number of observations"
  nobs  = size(Y,2);
  
  % Initialize output
  S.Zm  = NaN(m, nobs);       % Z_t | t-1 (prior)
  S.Vm  = NaN(m, m, nobs);    % V_t | t-1 (prior)
  S.ZmU = NaN(m, nobs+1);     % Z_t | t (posterior/updated)
  S.VmU = NaN(m, m, nobs+1);  % V_t | t (posterior/updated)
  S.loglik = 0;

%% SET INITIAL VALUES ----------------------------------------------------
  Zu = Z_0;  % Z_0|0 (In below loop, Zu gives Z_t | t)
  Vu = V_0;  % V_0|0 (In below loop, Vu guvse V_t | t)
  
  % Store initial values
  S.ZmU(:,1)    = Zu;
  S.VmU(:,:,1)  = Vu;

%% KALMAN FILTER PROCEDURE ----------------------------------------------
  for t = 1:nobs
      %%% CALCULATING PRIOR DISTIBUTION----------------------------------
      
      % Use transition eqn to create prior estimate for factor
      % i.e. Z = Z_t|t-1
      Z   = A * Zu;
      
      % Prior covariance matrix of Z (i.e. V = V_t|t-1)
      %   Var(Z) = Var(A*Z + u_t) = Var(A*Z) + Var(\epsilon) = 
      %   A*Vu*A' + Q
      V   = A * Vu* A' + Q; 
      V   =  0.5 * (V+V');  % Trick to make symmetric
      
      %%% CALCULATING POSTERIOR DISTRIBUTION ----------------------------
       
      % Removes missing series: These are removed from Y, C, and R
      [Y_t, C_t, R_t, ~] = MissData(Y(:,t), C, R); 

      % Check if y_t contains no data. If so, replace Zu and Vu with prior.
      if isempty(Y_t)
          Zu = Z;
          Vu = V;
      else  
          % Steps for variance and population regression coefficients:
          % Var(c_t*Z_t + e_t) = c_t Var(A) c_t' + Var(u) = c_t*V *c_t' + R
          VC  = V * C_t';  
          iF  = inv(C_t * VC + R_t);
          
          % Matrix of population regression coefficients (QuantEcon eqn #4)
          VCF = VC*iF;  

          % Gives difference between actual and predicted observation
          % matrix values
          innov  = Y_t - C_t*Z;
          
          % Update estimate of factor values (posterior)
          Zu  = Z  + VCF * innov;
          
          % Update covariance matrix (posterior) for time t
          Vu  = V  - VCF * VC';
          Vu   =  0.5 * (Vu+Vu'); % Approximation trick to make symmetric
          
          % Update log likelihood 
          S.loglik = S.loglik + 0.5*(log(det(iF))  - innov'*iF*innov);
      end
      
      %%% STORE OUTPUT----------------------------------------------------
      
      % Store covariance and observation values for t-1 (priors)
      S.Zm(:,t)   = Z;
      S.Vm(:,:,t) = V;

      % Store covariance and state values for t (posteriors)
      % i.e. Zu = Z_t|t   & Vu = V_t|t
      S.ZmU(:,t+1)    = Zu;
      S.VmU(:,:,t+1)  = Vu;
  end 
  
  % Store Kalman gain k_t
  if isempty(Y_t)
      S.k_t = zeros(m,m);
  else
      S.k_t = VCF * C_t;
  end
end
