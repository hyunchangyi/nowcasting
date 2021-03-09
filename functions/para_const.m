function Res = para_const(X, P, lag)
%para_const()    Implements Kalman filter for "News_DFM.m"
%
%  Syntax:
%    Res = para_const(X,P,lag)
%
%  Description:
%    para_const() implements the Kalman filter for the news calculation
%    step. This procedure smooths and fills in missing data for a given 
%    data matrix X. In contrast to runKF(), this function is used when
%    model parameters are already estimated.
%
%  Input parameters:
%    X: Data matrix. 
%    P: Parameters from the dynamic factor model.
%    lag: Number of lags
%
%  Output parameters:
%    Res [struc]: A structure containing the following:
%      Res.Plag: Smoothed factor covariance for transition matrix
%      Res.P:    Smoothed factor covariance matrix
%      Res.X_sm: Smoothed data matrix
%      Res.F:    Smoothed factors
%    


% Kalman filter with specified paramaters 
% written for 
% "MAXIMUM LIKELIHOOD ESTIMATION OF FACTOR MODELS ON DATA SETS WITH
% ARBITRARY PATTERN OF MISSING DATA."
% by Marta Banbura and Michele Modugno 

%% Set model parameters and data preparation

% Set model parameters
Z_0 = P.Z_0;
V_0 = P.V_0;
A = P.A;
C = P.C;
Q = P.Q;
R = P.R;
Mx = P.Mx;
Wx = P.Wx;

% Prepare data
[T,~] = size(X);

% Standardise x
Y = ((X-repmat(Mx,T,1))./repmat(Wx,T,1))';

%% Apply Kalman filter and smoother
% See runKF() for details about FIS and SKF

Sf = SKF(Y, A, C, Q, R, Z_0, V_0);  % Kalman filter

Ss = FIS(A, Sf);  % Smoothing step

%% Calculate parameter output
Vs = Ss.VmT(:,:,2:end);  % Smoothed factor covariance for transition matrix
Vf = Sf.VmU(:,:,2:end);  % Filtered factor posterior covariance
Zsmooth = Ss.ZmT;        % Smoothed factors
Vsmooth = Ss.VmT;        % Smoothed covariance values

Plag{1} = Vs;

for jk = 1:lag
    for jt = size(Plag{1},3):-1:lag+1
        As = Vf(:,:,jt-jk)*A'*pinv(A*Vf(:,:,jt-jk)*A'+Q);
        Plag{jk+1}(:,:,jt) = As*Plag{jk}(:,:,jt);
    end
end

% Prepare data for output
Zsmooth=Zsmooth';
x_sm = Zsmooth(2:end,:)*C';  % Factors to series representation
X_sm = repmat(Wx,T,1).*x_sm+repmat(Mx,T,1);  % Standardized to unstandardized

%--------------------------------------------------------------------------
%   Loading the structure with the results
%--------------------------------------------------------------------------
Res.Plag = Plag;
Res.P = Vsmooth;
Res.X_sm = X_sm;  
Res.F = Zsmooth(2:end,:); 

end
