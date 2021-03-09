function update_nowcast(X_old,X_new,Time,Spec,Res,series,period,vintage_old,vintage_new)


if ~isnumeric(vintage_old)
    vintage_old = datenum(vintage_old,'yyyy-mm-dd');
end
if ~isnumeric(vintage_new)
    vintage_new = datenum(vintage_new,'yyyy-mm-dd');
end

% Make sure datasets are the same size
N = size(X_new,2);
T_old = size(X_old,1); T_new = size(X_new,1);
if T_new > T_old
    X_old = [X_old; NaN(T_new-T_old,N)];
end
% append 1 year (12 months) of data to each dataset to allow for
% forecasting at different horizons
X_old = [X_old; NaN(12,N)];
X_new = [X_new; NaN(12,N)];

[y, m, d] = datevec(Time(end));

Time = [Time; datenum(y,(m+1:m+12)',d)];

i_series = find(strcmp(series,Spec.SeriesID));

series_name = Spec.SeriesName{i_series};
freq        = Spec.Frequency{i_series};

switch freq
    case 'm'
        [y,m] = strtok(period,freq);
        y = str2double(y);
        m = str2double(strrep(m,freq,''));
        d = 1;
        t_nowcast = find(Time==datenum(y,m,d));
    case 'q'
        [y,q] = strtok(period,freq);
        y = str2double(y);
        q = str2double(strrep(q,freq,''));
        m = 3*q;
        d = 1;
        t_nowcast = find(Time==datenum(y,m,d));
end

if isempty(t_nowcast)
    error('Period is out of nowcasting horizon (up to one year ahead).');
end



% Update nowcast for target variable 'series' (i) at horizon 'period' (t)
%   > Relate nowcast update into news from data releases:
%     a. Compute the impact from data revisions
%     b. Compute the impact from new data releases


X_rev = X_new;  
X_rev(isnan(X_old)) = NaN;  

% Compute news --------------------------------------------------------

% Compute impact from data revisions
[y_old] = News_DFM(X_old,X_rev,Res,t_nowcast,i_series);

% Compute impact from data releases
[y_rev,y_new,~,actual,forecast,weight] = News_DFM(X_rev,X_new,Res,t_nowcast,i_series);

% Display output
fprintf('\n\n\n');
fprintf('Nowcast Update: %s \n', datestr(vintage_new, 'mmmm dd, yyyy'))
fprintf('Nowcast for %s (%s), %s \n',Spec.SeriesName{i_series},Spec.UnitsTransformed{i_series},datestr(Time(t_nowcast),'YYYY:QQ'));

if(isempty(forecast))  % Only display table output if a forecast is made
    fprintf('\n  No forecast was made.\n')
else

    impact_revisions = y_rev - y_old;      % Impact from revisions
    news = actual - forecast;              % News from releases
    impact_releases = weight .* (news);    % Impact of releases

    % Store results
    news_table = array2table([forecast, actual, weight, impact_releases],...
                            'VariableNames', {'Forecast', 'Actual', 'Weight', 'Impact'},...
                            'RowNames', Spec.SeriesID);

    % Select only series with updates
    data_released = any(isnan(X_old) & ~isnan(X_new), 1);  

    % Display the impact decomposition
    fprintf('\n  Nowcast Impact Decomposition \n')
    fprintf('  Note: The displayed output is subject to rounding error\n\n')
    fprintf('                  %s nowcast:                  %5.2f\n', ...
            datestr(vintage_old, 'mmm dd'), y_old)
    fprintf('      Impact from data revisions:      %5.2f\n', impact_revisions)
    fprintf('       Impact from data releases:      %5.2f\n', sum(news_table.Impact,'omitnan'))
    fprintf('                                     +_________\n')
    fprintf('                    Total impact:      %5.2f\n', ...
            impact_revisions + sum(news_table.Impact,'omitnan'))
    fprintf('                  %s nowcast:                  %5.2f\n\n', ...
        datestr(vintage_new, 'mmm dd'), y_new)

    % Display the table output
    fprintf('\n  Nowcast Detail Table \n\n')
    disp(news_table(data_released, :))

end

% Output results to structure



end