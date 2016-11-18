function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
m = size(X,1);
idx = zeros( m , 1);

A = zeros( m , K );

for i=1:K
    
   A(:, i) = sqrt( sum( (X -  ones( size(X) )*diag( centroids(i,:) ) ).^2 , 2 ) );  
end

[b, idx] = min(A , [] , 2); 

end

