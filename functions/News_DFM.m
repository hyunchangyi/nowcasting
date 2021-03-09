function [y_old,y_new,singlenews,actual,forecast,weight,t_miss,v_miss,innov] = News_DFM(X_old,X_new,Res,t_fcst,v_news)
%News_DFM()    Calculates changes in news
%             
%  Syntax:
%  [y_old, y_new, singlenews, actual, fore, weight ,t_miss, v_miss, innov] = ...
%    News_DFM(X_old, X_new, Q, t_fcst, v_news) 
%
%  Description:
%   News DFM() inputs two datasets, DFM parameters, target time index, and 
%   target variable index. The function then produces Nowcast updates and
%   decomposes the changes into news.
%
%  Input Arguments:
%    X_old:  Old data matrix (old vintage)
%    X_new:  New data matrix (new vintage)
%    Res:    DFM() output results (see DFM for more details)
%    t_fcst: Index for target time
%    v_news: Index for target variable 
%
%  Output Arguments:
%    y_old:       Old nowcast
%    y_new:       New nowcast
%    single_news: News for each data series
%    actual:      Observed series release values
%    fore:        Forecasted series values
%    weight:      News weight
%    t_miss:      Time index for data releases
%    v_miss:      Series index for data releases
%    innov:       Difference between observed and predicted series values ("innovation")


%(X_rev,X_new,Res,t_nowcast,i_series) => (X_old,X_new,Res,t_fcst,v_news)
X_old = X_rev;
t_fcst = t_nowcast;
v_news = i_series;

%% Initialize variables

r = size(Res.C,2);
[~, N] = size(X_new);
singlenews = zeros(1,N);  % Initialize news vector (will store news for each series)

%% NO FORECAST CASE: Already values for variables v_news at time t_fcst

if ~isnan(X_new(t_fcst,v_news)) 
    Res_old = para_const(X_old, Res, 0);  % Apply Kalman filter for old data
    

    for i=1:size(v_news,2)      % Loop for each target variable
        % (Observed value) - (predicted value)
        singlenews(:,v_news(i)) = X_new(t_fcst,v_news(i)) ...
                                    - Res_old.X_sm(t_fcst,v_news(i));
        
        % Set predicted and observed y values
        y_old(1,i) = Res_old.X_sm(t_fcst,v_news(i));
        y_new(1,i) = X_new(t_fcst,v_news(i));
    end
    
    % Forecast-related output set to empty
    actual=[];forecast=[];weight=[];t_miss=[];v_miss=[];innov=[];


else
    %% FORECAST CASE (these are broken down into (A) and (B))

    % Initialize series mean/standard deviation respectively
    Mx = Res.Mx;
    Wx = Res.Wx;
    
    % Calculate indicators for missing values (1 if missing, 0 otherwise)
    miss_old=isnan(X_old);
    miss_new=isnan(X_new);
    
    % Indicator for missing--combine above information to single matrix where:
    % (i) -1: Value is in the old data, but missing in new data
    % (ii) 1: Value is in the new data, but missing in old data 
    % (iii) 0: Values are missing from/available in both datasets
    i_miss = miss_old - miss_new;
    
    % Time/variable indicies where case (b) is true
    [t_miss,v_miss]=find(i_miss==1);
    
    %% FORECAST SUBCASE (A): NO NEW INFORMATION
    
    if isempty(v_miss)
        % Fill in missing variables using a Kalman filter
        Res_old = para_const(X_old, Res, 0);
        Res_new = para_const(X_new, Res, 0);
        
        % Set predicted and observed y values. New y value is set to old
        y_old = Res_old.X_sm(t_fcst,v_news);
        y_new = y_old;
        % y_new = Res_new.X_sm(t_fcst,v_news);
        
        % No news, so nothing returned for news-related output
        groupnews=[];singlenews=[];gain=[];gainSer=[];
        actual=[];forecast=[];weight=[];t_miss=[];v_miss=[];innov=[];
    else
    %----------------------------------------------------------------------
    %     v_miss=[1:size(X_new,2)]';
    %     t_miss=t_miss(1)*ones(size(X_new,2),1);
    %----------------------------------------------------------------------
    %% FORECAST SUBCASE (B): NEW INFORMATION
   
    % Difference between forecast time and new data time
    lag = t_fcst-t_miss;
    
    % Gives biggest time interval between forecast and new data
    k = max([abs(lag); max(lag)-min(lag)]);
    
    C = Res.C;   % Observation matrix
    R = Res.R';  % Covariance for observation matrix residuals
    
    % Number of new events
    n_news = size(lag,1);
    
    % Smooth old dataset
    Res_old = para_const(X_old, Res, k);
    Plag = Res_old.Plag;
    
    % Smooth new dataset
    Res_new = para_const(X_new, Res, 0);
    
    % Subset for target variable and forecast time
    y_old = Res_old.X_sm(t_fcst,v_news);
    y_new = Res_new.X_sm(t_fcst,v_news);
    
    
    
    P = Res_old.P(:,:,2:end);
    P1=[];  % Initialize projection onto updates
    
    % Cycle through total number of updates
    for i=1:n_news
        h = abs(t_fcst-t_miss(i));
        m = max([t_miss(i) t_fcst]);
        
        % If location of update is later than the forecasting date
        if t_miss(i)>t_fcst
            Pp=Plag{h+1}(:,:,m);  %P(1:r,h*r+1:h*r+r,m)';
        else
            Pp=Plag{h+1}(:,:,m)';  %P(1:r,h*r+1:h*r+r,m);
        end
        P1=[P1 Pp*C(v_miss(i),1:r)'];  % Projection on updates
    end
    
    for i=1:size(t_miss,1)
        % Standardize predicted and observed values
        X_new_norm = (X_new(t_miss(i),v_miss(i)) - Mx(v_miss(i)))./Wx(v_miss(i));
        X_sm_norm = (Res_old.X_sm(t_miss(i),v_miss(i))- Mx(v_miss(i)))./Wx(v_miss(i));
        
        % Innovation: Gives [observed] data - [predicted data]
        innov(i)= X_new_norm - X_sm_norm;          
    end
    
    ins=size(innov,2);
    P2=[];
    p2=[];
    
    % Gives non-standardized series weights
    for i=1:size(lag,1)
        for j=1:size(lag,1)
            h=abs(lag(i)-lag(j));
            m=max([t_miss(i),t_miss(j)]);
            
            if t_miss(j)>t_miss(i)
                Pp=Plag{h+1}(:,:,m); %P(1:r,h*r+1:(h+1)*r,m)';
            else
                Pp=Plag{h+1}(:,:,m)'; %P(1:r,h*r+1:(h+1)*r,m);
            end
            if v_miss(i)==v_miss(j) & t_miss(i)~=t_miss(j)
                WW(v_miss(i),v_miss(j))=0;
            else
                WW(v_miss(i),v_miss(j))=R(v_miss(i),v_miss(j));
            end
            p2=[p2 C(v_miss(i),1:r)*Pp*C(v_miss(j),1:r)'+WW(v_miss(i),v_miss(j))];
        end
        P2=[P2;p2];
        p2=[];
    end
    
    clear temp
    for i=1:size(v_news,2)      % loop on v_news
        % Convert to real units (unstadardized data)
        totnews(1,i) = Wx(v_news(i))*C(v_news(i),1:r)*P1*inv(P2)*innov';
        temp(1,:,i) = Wx(v_news(i))*C(v_news(i),1:r)*P1*inv(P2).*innov;
        gain(:,:,i) = Wx(v_news(i))*C(v_news(i),1:r)*P1*inv(P2);
    end
    
    % Initialize output objects
    singlenews = NaN(max(t_miss)-min(t_miss)+1,N); %,size(v_news,2)
    actual     = NaN(N,1);  % Actual forecasted values
    forecast   = NaN(N,1);  % Forecasted values
    weight     = NaN(N,1,size(v_news,2));
    
    % Fill in output values 
    for i=1:size(innov,2)
        actual(v_miss(i),1) = X_new(t_miss(i),v_miss(i));  
        forecast(v_miss(i),1) = Res_old.X_sm(t_miss(i),v_miss(i));
        
        for j=1:size(v_news,2)
            singlenews(t_miss(i)-min(t_miss)+1,v_miss(i),j) = temp(1,i,j);
            weight(v_miss(i),:,j) = gain(:,i,j)/Wx(v_miss(i));
        end
    end
    
    singlenews = sum(singlenews,1);      % Returns total news

    
    [v_miss, ~, ~] = unique(v_miss);  
    
end

end

end