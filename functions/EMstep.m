function  [C_new, R_new, A_new, Q_new, Z_0, V_0, loglik] = EMstep(y, A, C, Q, R, Z_0, V_0, r,p,R_mat,q,nQ,i_idio,blocks)
%EMstep    Applies EM algorithm for parameter reestimation
%
%  Syntax:
%    [C_new, R_new, A_new, Q_new, Z_0, V_0, loglik]
%    = EMstep(y, A, C, Q, R, Z_0, V_0, r, p, R_mat, q, nQ, i_idio, blocks)
%
%  Description:
%    EMstep reestimates parameters based on the Estimation Maximization (EM)
%    algorithm. This is a two-step procedure:
%    (1) E-step: the expectation of the log-likelihood is calculated using
%        previous parameter estimates.
%    (2) M-step: Parameters are re-estimated through the maximisation of
%        the log-likelihood (maximize result from (1)).
%
%    See "Maximum likelihood estimation of factor models on data sets with
%    arbitrary pattern of missing data" for details about parameter
%    derivation (Banbura & Modugno, 2010). This procedure is in much the
%    same spirit.
%
%  Input:
%    y:      Series data
%    A:      Transition matrix
%    C:      Observation matrix
%    Q:      Covariance for transition equation residuals
%    R:      Covariance for observation matrix residuals
%    Z_0:    Initial values of factors
%    V_0:    Initial value of factor covariance matrix
%    r:      Number of common factors for each block (e.g. vector [1 1 1 1])
%    p:      Number of lags in transition equation
%    R_mat:  Estimation structure for quarterly variables (i.e. "tent")
%    q:      Constraints on loadings
%    nQ:     Number of quarterly series
%    i_idio: Indices for monthly variables
%    blocks: Block structure for each series (i.e. for a series, the structure
%            [1 0 0 1] indicates loadings on the first and fourth factors)
%
%  Output:
%    C_new: Updated observation matrix
%    R_new: Updated covariance matrix for residuals of observation matrix
%    A_new: Updated transition matrix
%    Q_new: Updated covariance matrix for residuals for transition matrix
%    Z_0:   Initial value of state
%    V_0:   Initial value of covariance matrix
%    loglik: Log likelihood
%
% References:
%   "Maximum likelihood estimation of factor models on data sets with
%   arbitrary pattern of missing data" by Banbura & Modugno (2010).
%   Abbreviated as BM2010
%
%

%% Initialize preliminary values

% Store series/model values
[n, T] = size(y);
nM = n - nQ;  % Number of monthly series
pC = size(R_mat,2);
ppC = max(p,pC);
num_blocks = size(blocks,2);  % Number of blocks

%% ESTIMATION STEP: Compute the (expected) sufficient statistics for a single
%Kalman filter sequence

% Running the Kalman filter and smoother with current parameters
% Note that log-liklihood is NOT re-estimated after the runKF step: This
% effectively gives the previous iteration's log-likelihood
% For more information on output, see runKF
[Zsmooth, Vsmooth, VVsmooth, loglik] = runKF(y, A, C, Q, R, Z_0, V_0);


%% MAXIMIZATION STEP (TRANSITION EQUATION)
% See (Banbura & Modugno, 2010) for details.

% Initialize output
A_new = A;
Q_new = Q;
V_0_new = V_0;

%%% 2A. UPDATE FACTOR PARAMETERS INDIVIDUALLY ----------------------------

for i = 1:num_blocks  % Loop for each block: factors are uncorrelated

    % SETUP INDEXING
    r_i = r(i);  % r_i = 1 if block is loaded
    rp = r_i*p;
    rp1 = sum(r(1:i-1))*ppC;
    b_subset = rp1+1:rp1+rp;  % Subset blocks: Helps for subsetting Zsmooth, Vsmooth
    t_start = rp1+1;          % Transition matrix factor idx start
    t_end = rp1+r_i*ppC;      % Transition matrix factor idx end



    % ESTIMATE FACTOR PORTION OF Q, A
    % Note: EZZ, EZZ_BB, EZZ_FB are parts of equations 6 and 8 in BM 2010

    % E[f_t*f_t' | Omega_T]
    EZZ = Zsmooth(b_subset, 2:end) * Zsmooth(b_subset, 2:end)'...
        +sum(Vsmooth(b_subset, b_subset, 2:end) ,3);

    % E[f_{t-1}*f_{t-1}' | Omega_T]
    EZZ_BB = Zsmooth(b_subset, 1:end-1)*Zsmooth(b_subset, 1:end-1)'...
            +sum(Vsmooth(b_subset, b_subset, 1:end-1), 3);

    % E[f_t*f_{t-1}' | Omega_T]
    EZZ_FB = Zsmooth(b_subset, 2:end)*Zsmooth(b_subset, 1:end-1)'...
        +sum(VVsmooth(b_subset, b_subset, :), 3);

    % Select transition matrix/covariance matrix for block i
    A_i = A(t_start:t_end, t_start:t_end);
    Q_i = Q(t_start:t_end, t_start:t_end);

    % Equation 6: Estimate VAR(p) for factor
    A_i(1:r_i,1:rp) = EZZ_FB(1:r_i,1:rp) * inv(EZZ_BB(1:rp,1:rp));

    % Equation 8: Covariance matrix of residuals of VAR
    Q_i(1:r_i,1:r_i) = (EZZ(1:r_i,1:r_i) - A_i(1:r_i,1:rp)* EZZ_FB(1:r_i,1:rp)') / T;

    % Place updated results in output matrix
    A_new(t_start:t_end, t_start:t_end) = A_i;
    Q_new(t_start:t_end, t_start:t_end) = Q_i;
    V_0_new(t_start:t_end, t_start:t_end) =...
        Vsmooth(t_start:t_end, t_start:t_end,1);
end

%%
%%% 2B. UPDATING PARAMETERS FOR IDIOSYNCRATIC COMPONENT ------------------

rp1 = sum(r)*ppC;           % Col size of factor portion
niM = sum(i_idio(1:nM));    % Number of monthly values
t_start = rp1+1;            % Start of idiosyncratic component index
i_subset = t_start:rp1+niM; % Gives indices for monthly idiosyncratic component values


% Below 3 estimate the idiosyncratic component (for eqns 6, 8 BM 2010)

% E[f_t*f_t' | \Omega_T]
EZZ = diag(diag(Zsmooth(t_start:end, 2:end) * Zsmooth(t_start:end, 2:end)'))...
    + diag(diag(sum(Vsmooth(t_start:end, t_start:end, 2:end), 3)));

% E[f_{t-1}*f_{t-1}' | \Omega_T]
EZZ_BB = diag(diag(Zsmooth(t_start:end, 1:end-1)* Zsmooth(t_start:end, 1:end-1)'))...
       + diag(diag(sum(Vsmooth(t_start:end, t_start:end, 1:end-1), 3)));

% E[f_t*f_{t-1}' | \Omega_T]
EZZ_FB = diag(diag(Zsmooth(t_start:end, 2:end)*Zsmooth(t_start:end, 1:end-1)'))...
       + diag(diag(sum(VVsmooth(t_start:end, t_start:end, :), 3)));

A_i = EZZ_FB * diag(1./diag((EZZ_BB)));  % Equation 6
Q_i = (EZZ - A_i*EZZ_FB') / T;           % Equation 8

% Place updated results in output matrix
A_new(i_subset, i_subset) = A_i(1:niM,1:niM);
Q_new(i_subset, i_subset) = Q_i(1:niM,1:niM);
V_0_new(i_subset, i_subset) = diag(diag(Vsmooth(i_subset, i_subset, 1)));


%% 3 MAXIMIZATION STEP (observation equation)

%%% INITIALIZATION AND SETUP ----------------------------------------------
Z_0 = Zsmooth(:,1); %zeros(size(Zsmooth,1),1); %

% Set missing data series values to 0
nanY = isnan(y);
y(nanY) = 0;

% LOADINGS
C_new = C;

% Blocks
bl = unique(blocks,'rows');  % Gives unique loadings
n_bl = size(bl,1);           % Number of unique loadings

% Initialize indices: These later help with subsetting
bl_idxM = [];  % Indicator for monthly factor loadings
bl_idxQ = [];  % Indicator for quarterly factor loadings
R_con = [];    % Block diagonal matrix giving monthly-quarterly aggreg scheme
q_con = [];

% Loop through each block
for i = 1:num_blocks
    bl_idxQ = [bl_idxQ repmat(bl(:,i),1,r(i)*ppC)];
    bl_idxM = [bl_idxM repmat(bl(:,i),1,r(i)) zeros(n_bl,r(i)*(ppC-1))];
    R_con = blkdiag(R_con, kron(R_mat,eye(r(i))));
    q_con = [q_con;zeros(r(i)*size(R_mat,1),1)];
end

% Indicator for monthly/quarterly blocks in observation matrix
bl_idxM = logical(bl_idxM);
bl_idxQ = logical(bl_idxQ);

i_idio_M = i_idio(1:nM);            % Gives 1 for monthly series
n_idio_M = length(find(i_idio_M));  % Number of monthly series
c_i_idio = cumsum(i_idio);          % Cumulative number of monthly series

for i = 1:n_bl  % Loop through unique loadings (e.g. [1 0 0 0], [1 1 0 0])

    bl_i = bl(i,:);
    rs = sum(r(logical(bl_i)));                    % Total num of blocks loaded
    idx_i = find(ismember(blocks, bl_i, 'rows'));  % Indices for bl_i
    idx_iM = idx_i(idx_i<nM+1);                    % Only monthly
    n_i = length(idx_iM);                          % Number of monthly series

    % Initialize sums in equation 13 of BGR 2010
    denom = zeros(n_i*rs,n_i*rs);
    nom = zeros(n_i,rs);

    % Stores monthly indicies. These are done for input robustness
    i_idio_i = i_idio_M(idx_iM);
    i_idio_ii = c_i_idio(idx_iM);
    i_idio_ii = i_idio_ii(i_idio_i);

    %%% UPDATE MONTHLY VARIABLES: Loop through each period ----------------
    for t = 1:T
        Wt = diag(~nanY(idx_iM, t));  % Gives selection matrix (1 for nonmissing values)

        denom = denom +...  % E[f_t*t_t' | Omega_T]
                kron(Zsmooth(bl_idxM(i, :), t+1) * Zsmooth(bl_idxM(i, :), t+1)' + ...
                Vsmooth(bl_idxM(i, :), bl_idxM(i, :), t+1), Wt);

        nom = nom + ...  E[y_t*f_t' | \Omega_T]
              y(idx_iM, t) * Zsmooth(bl_idxM(i, :), t+1)' - ...
              Wt(:, i_idio_i) * (Zsmooth(rp1 + i_idio_ii, t+1) * ...
              Zsmooth(bl_idxM(i, :), t+1)' + ...
              Vsmooth(rp1 + i_idio_ii, bl_idxM(i, :), t+1));
    end
    
    vec_C = inv(denom)*nom(:);  % Eqn 13 BGR 2010

    % Place updated monthly results in output matrix
    C_new(idx_iM,bl_idxM(i,:)) = reshape(vec_C, n_i, rs);

   %%% UPDATE QUARTERLY VARIABLES -----------------------------------------

   idx_iQ = idx_i(idx_i > nM);  % Index for quarterly series
   rps = rs * ppC;

   % Monthly-quarterly aggregation scheme
   R_con_i = R_con(:,bl_idxQ(i,:));
   q_con_i = q_con;

   no_c = ~(any(R_con_i,2));
   R_con_i(no_c,:) = [];
   q_con_i(no_c,:) = [];

   % Loop through quarterly series in loading. This parallels monthly code
   for j = idx_iQ'
       % Initialization
       denom = zeros(rps,rps);
       nom = zeros(1,rps);

       idx_jQ = j-nM;  % Ordinal position of quarterly variable
       % Loc of factor structure corresponding to quarterly var residuals
       i_idio_jQ = (rp1 + n_idio_M + 5*(idx_jQ-1)+1:rp1+ n_idio_M + 5*idx_jQ);

       % Place quarterly values in output matrix
       V_0_new(i_idio_jQ, i_idio_jQ) = Vsmooth(i_idio_jQ, i_idio_jQ,1);
       A_new(i_idio_jQ(1), i_idio_jQ(1)) = A_i(i_idio_jQ(1)-rp1, i_idio_jQ(1)-rp1);
       Q_new(i_idio_jQ(1), i_idio_jQ(1)) = Q_i(i_idio_jQ(1)-rp1, i_idio_jQ(1)-rp1);

       for t=1:T
           Wt = diag(~nanY(j,t));  % Selection matrix for quarterly values

           % Intermediate steps in BGR equation 13
           denom = denom + ...
                   kron(Zsmooth(bl_idxQ(i,:), t+1) * Zsmooth(bl_idxQ(i,:), t+1)'...
                 + Vsmooth(bl_idxQ(i,:), bl_idxQ(i,:), t+1), Wt);
           nom = nom + y(j, t)*Zsmooth(bl_idxQ(i,:), t+1)';
           nom = nom -...
                Wt * ([1 2 3 2 1] * Zsmooth(i_idio_jQ,t+1) * ...
                Zsmooth(bl_idxQ(i,:),t+1)'+...
                [1 2 3 2 1]*Vsmooth(i_idio_jQ,bl_idxQ(i,:),t+1));
       end

        C_i = inv(denom) * nom';
        C_i_constr = C_i - ...  % BGR equation 13
                     inv(denom) * R_con_i'*inv(R_con_i*inv(denom)*R_con_i') * (R_con_i*C_i-q_con_i);

        % Place updated values in output structure
        C_new(j,bl_idxQ(i,:)) = C_i_constr;
   end
end

%%
%%% 3B. UPDATE COVARIANCE OF RESIDUALS FOR OBSERVATION EQUATION -----------
% Initialize covariance of residuals of observation equation
R_new = zeros(n,n);
for t=1:T
    Wt = diag(~nanY(:,t));  % Selection matrix
    R_new = R_new + (y(:,t) - ...  % BGR equation 15
            Wt * C_new * Zsmooth(:, t+1)) * (y(:,t) - Wt*C_new*Zsmooth(:,t+1))'...
           + Wt*C_new*Vsmooth(:,:,t+1)*C_new'*Wt + (eye(n)-Wt)*R*(eye(n)-Wt);
end


R_new = R_new/T;
RR = diag(R_new); %RR(RR<1e-2) = 1e-2;
RR(i_idio_M) = 1e-04;  % Ensure non-zero measurement error. See Doz, Giannone, Reichlin (2012) for reference.
RR(nM+1:end) = 1e-04;
R_new = diag(RR);

end