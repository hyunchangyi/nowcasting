function [y,C,R,L]  = MissData(y,C,R)
%______________________________________________________________________
% PROC missdata                                                        
% PURPOSE: eliminates the rows in y & matrices Z, G that correspond to     
%          missing data (NaN) in y                                                                                  
% INPUT    y             vector of observations at time t    
%          S             KF system matrices (structure)
%                        must contain Z & G
% OUTPUT   y             vector of observations (reduced)     
%          Z G           KF system matrices     (reduced)     
%          L             To restore standard dimensions     
%                        where # is the nr of available data in y
%______________________________________________________________________
  
  % Returns 1 for nonmissing series
  ix = ~isnan(y);
  
  % Index for columns with nonmissing variables
  e  = eye(size(y,1));
  L  = e(:,ix);

  % Removes missing series
  y  = y(ix);
  
  % Removes missing series from observation matrix
  C  =  C(ix,:);  
  
  % Removes missing series from transition matrix
  R  =  R(ix,ix);

end