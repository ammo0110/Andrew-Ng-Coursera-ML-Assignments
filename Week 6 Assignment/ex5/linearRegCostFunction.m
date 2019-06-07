function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% X is m * (n+1) matrix
% Calculate hypothesis for all training sets
h = X*theta;	% h is an m * 1 vector
diff = (h-y);
J = (diff'*diff) / (2*m);

% Add regularization term to J
% Slicing the first row from theta since we dont need bias terms for regularization
J += lambda * sum(theta([2:end], :).^2) / (2*m);

% Now calculating gradients
grad = (diff'*X) / (m);	% grad is 1 * (n+1) vector
% Transpose grad
grad = grad';
% Add regularization terms
n = size(theta, 2);
% Slicing and appending zeros to theta vector since we don't need to regularize j=0
grad += (lambda/m) * [zeros(1, n) ; theta([2:end], :)]; 





% =========================================================================

grad = grad(:);

end
