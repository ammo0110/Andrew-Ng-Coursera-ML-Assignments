function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

h = X*theta; % The hypothesis function vector of m*1

t = h-y;  % Compute the difference simultaneously for all training examples

J = 0.5*(t'*t)/m; % t'*t will give the square sum of all elements of t


% =========================================================================

end
