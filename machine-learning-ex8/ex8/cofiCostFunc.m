function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0.5 * sum ( sum ( R.*(X*Theta' - Y).^2 ) );

J = J + 0.5*lambda* ( norm( Theta, 'fro' )^2 + norm( X, 'fro' )^2 );  

X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


for k=1:num_features
   X_grad(:, k) = ( R.*(X*Theta' - Y) )*Theta(:, k); 
   Theta_grad(:, k) = transpose( X(:, k)'*( R.*(X*Theta' - Y) ) ); 
end

X_grad = X_grad + lambda*X;
Theta_grad = Theta_grad + lambda*Theta;

grad = [X_grad(:); Theta_grad(:)];

end
