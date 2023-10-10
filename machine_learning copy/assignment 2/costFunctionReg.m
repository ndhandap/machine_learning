function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

for j = 1:size(theta),	
	for i =1:m,
		x = X(i,:);
		h_theta = sigmoid(theta'*x')          %this is not an array, this is just a single value
		J = J + (((-1*y(i)*log(h_theta)) - ((1-y(i))*log(1-h_theta)))/(m*length(theta))) + ((lambda/(2*m*m))*theta(j)^2); 
		grad(j) = grad(j) + (((h_theta - y(i))*X(i,j))/m) ;

	end
	if j == 1
		grad(j) = grad(j);
	else
		grad(j) = grad(j) +  ((lambda/(m))*theta(j));
	endif
end


%grad(1) = theta(1)



% =============================================================

end
