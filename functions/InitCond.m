function [ A, C, Q, R, Z_0, V_0] = InitCond(x,r,p,blocks,optNaN,Rcon,q,nQ,i_idio)

pC = size(Rcon,2);  % Gives 'tent' structure size (quarterly to monthly)
ppC = max(p,pC);
n_b = size(blocks,2);  % Number of blocks

OPTS.disp=0;  % Turns off diagnostic information for eigenvalue computation
[xBal,indNaN] = remNaNs_spline(x,optNaN);  % Spline without NaNs

[T,N] = size(xBal);  % Time T series number N
nM = N-nQ;           % Number of monthly series

xNaN = xBal;
xNaN(indNaN) = nan;  % Set missing values equal to NaNs
res = xBal;          % Spline output equal to res: Later this is used for residuals
resNaN = xNaN;       % Later used for residuals

% Initialize model coefficient output
C = [];
A = [];
Q = [];
V_0 = [];

% Set the first observations as NaNs: For quarterly-monthly aggreg. scheme
indNaN(1:pC-1,:) = true;

for i = 1:n_b  % Loop for each block

    r_i = r(i);  % r_i = 1 when block is loaded

    %% Observation equation -----------------------------------------------

    C_i = zeros(N,r_i*ppC);     % Initialize state variable matrix helper
    idx_i = find(blocks(:,i));  % Returns series index loading block i
    idx_iM = idx_i(idx_i<nM+1); % Monthly series indicies for loaded blocks
    idx_iQ = idx_i(idx_i>nM);   % Quarterly series indicies for loaded blocks



    % Returns eigenvector v w/largest eigenvalue d
    [v, d] = eigs(cov(res(:,idx_iM)), r_i, 'lm');

    % Flip sign for cleaner output. Gives equivalent results without this section
    if(sum(v) < 0)
        v = -v;
    end

    % For monthly series with loaded blocks (rows), replace with eigenvector
    % This gives the loading
    C_i(idx_iM,1:r_i) = v;
    f = res(:,idx_iM)*v;  % Data projection for eigenvector direction
    F = [];

    % Lag matrix using loading. This is later used for quarterly series
    for kk = 0:max(p+1,pC)-1
        F = [F f(pC-kk:end-kk,:)];
    end

    Rcon_i = kron(Rcon,eye(r_i));  % Quarterly-monthly aggregation scheme
    q_i = kron(q,zeros(r_i,1));

    % Produces projected data with lag structure (so pC-1 fewer entries)
    ff = F(:, 1:r_i*pC);

    for j = idx_iQ'      % Loop for quarterly variables

        % For series j, values are dropped to accommodate lag structure
        xx_j = resNaN(pC:end,j);

        if sum(~isnan(xx_j)) < size(ff,2)+2
            xx_j = res(pC:end,j);  % Replaces xx_j with spline if too many NaNs

        end

        ff_j = ff(~isnan(xx_j),:);
        xx_j = xx_j(~isnan(xx_j));

        iff_j = inv(ff_j'*ff_j);
        Cc = iff_j*ff_j'*xx_j;  % Least squares

        % Spline data monthly to quarterly conversion
        Cc = Cc - iff_j*Rcon_i'*inv(Rcon_i*iff_j*Rcon_i')*(Rcon_i*Cc-q_i);

        C_i(j,1:pC*r_i)=Cc';  % Place in output matrix
    end

    ff = [zeros(pC-1,pC*r_i);ff];  % Zeros in first pC-1 entries (replace dropped from lag)

    % Residual calculations
    res = res - ff*C_i';
    resNaN = res;
    resNaN(indNaN) = nan;

    C = [C C_i];  % Combine past loadings together


    %% Transition equation ------------------------------------------------

    z = F(:,1:r_i);            % Projected data (no lag)
    Z = F(:,r_i+1:r_i*(p+1));  % Data with lag 1

    A_i = zeros(r_i*ppC,r_i*ppC)';  % Initialize transition matrix

    A_temp = inv(Z'*Z)*Z'*z;  % OLS: gives coefficient value AR(p) process
    A_i(1:r_i,1:r_i*p) = A_temp';
    A_i(r_i+1:end,1:r_i*(ppC-1)) = eye(r_i*(ppC-1));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Q_i = zeros(ppC*r_i,ppC*r_i);
    e = z  - Z*A_temp;         % VAR residuals
    Q_i(1:r_i,1:r_i) = cov(e); % VAR covariance matrix

    initV_i = reshape(inv(eye((r_i*ppC)^2)-kron(A_i,A_i))*Q_i(:),r_i*ppC,r_i*ppC);

    % Gives top left block for the transition matrix
    A = blkdiag(A,A_i);
    Q = blkdiag(Q,Q_i);
    V_0 = blkdiag(V_0,initV_i);
end

eyeN = eye(N);  % Used inside observation matrix
eyeN(:,~i_idio) = [];

C=[C eyeN];
C = [C [zeros(nM,5*nQ); kron(eye(nQ),[1 2 3 2 1])]];  % Monthly-quarterly agreggation scheme
R = diag(var(resNaN,'omitnan'));  % Initialize covariance matrix for transition matrix


ii_idio = find(i_idio);    % Indicies for monthly variables
n_idio = length(ii_idio);  % Number of monthly variables
BM = zeros(n_idio);        % Initialize monthly transition matrix values
SM = zeros(n_idio);        % Initialize monthly residual covariance matrix values


for i = 1:n_idio;  % Loop for monthly variables
    % Set observation equation residual covariance matrix diagonal
    R(ii_idio(i),ii_idio(i)) = 1e-04;

    % Subsetting series residuals for series i
    res_i = resNaN(:,ii_idio(i));

    % Returns number of leading/ending zeros
    leadZero = max( find( (1:T)' == cumsum(isnan(res_i)) ) );
    endZero  = max( find( (1:T)' == cumsum(isnan(res_i(end:-1:1))) ) );

    % Truncate leading and ending zeros
    res_i = res(:,ii_idio(i));
    res_i(end-endZero + 1:end) = [];
    res_i(1:leadZero) = [];

    % Linear regression: AR 1 process for monthly series residuals
    BM(i,i) = inv(res_i(1:end-1)'*res_i(1:end-1))*res_i(1:end-1)'*res_i(2:end,:);
    SM(i,i) = cov(res_i(2:end)-res_i(1:end-1)*BM(i,i));  % Residual covariance matrix

end

Rdiag = diag(R);
sig_e = Rdiag(nM+1:N)/19;
Rdiag(nM+1:N) = 1e-04;
R = diag(Rdiag);  % Covariance for obs matrix residuals

% For BQ, SQ
rho0 = 0.1;
temp = zeros(5);
temp(1,1) = 1;

% Blocks for covariance matrices
SQ = kron(diag((1-rho0^2)*sig_e),temp);
BQ = kron(eye(nQ),[[rho0 zeros(1,4)];[eye(4),zeros(4,1)]]);
initViQ = reshape(inv(eye((5*nQ)^2)-kron(BQ,BQ))*SQ(:),5*nQ,5*nQ);
initViM = diag(1./diag(eye(size(BM,1))-BM.^2)).*SM;

% Output
A = blkdiag(A, BM, BQ);                % Observation matrix
Q = blkdiag(Q, SM, SQ);                % Residual covariance matrix (transition)
Z_0 = zeros(size(A,1),1);              % States
V_0 = blkdiag(V_0, initViM, initViQ);  % Covariance of states

end