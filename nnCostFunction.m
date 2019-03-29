%{
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
%
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%size(X)
a1 = [ones(m,1),X];
a2 = sigmoid(a1*Theta1');
a2 = [ones(m,1),a2];
a3 = sigmoid(a2*Theta2');
Y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
% fprintf('\nsize of Y is:'); size(Y)
J = (-1 / m) * sum(sum(Y.*log(a3) + (1 - Y).*log(1 - a3)));
%implementing regularization:
regTheta1 = Theta1(:,2:end);
regTheta2 = Theta2(:,2:end);
regTerm = (lambda/(2*m)) * (sum(sum(regTheta1.^2)) + sum(sum(regTheta2.^2)));
J += regTerm;
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
Del1 = zeros(size(Theta1));
Del2 = zeros(size(Theta2));
% fprintf('size of a1 is:');
% size(a1)
for i = 1:m
	a1t = X(i,:);	%1X400
	a2t = sigmoid([1,a1t]*Theta1');	%a2t = 1X25, Theta1 = 25X401
	a3t = sigmoid([1,a2t]*Theta2');	%a3t = 1X10, Theta2 = 10X26

	d3 = a3t-Y(i,:);	%is a 1X10 matrix
	d3 = d3';	%converted to a 10X1 vector
	% fprintf('\nsize of a2t:'); size(a2t)
	% fprintf('\nsize of d3:'); size(d3)
	% fprintf("\nsize of Theta2'*d3:"); temp = Theta2'*d3; size(temp)
	% fprintf('\nsize of sigmoidGradient([1,a2t]):'); size(sigmoidGradient([1;a2t']))
	d2 = (Theta2'*d3).*sigmoidGradient([1;a2t']);	%d2 = 26X1 matrix
	% fprintf('\nsize of d2:');
	% size(d2)
	%d2 = d2';	%d2 = 26X1

	Del1 = Del1 + d2(2:end)*[1 a1t];
	Del2 = Del2 + d3*[1 a2t];
endfor	

Theta1_grad += 1/m * Del1;
Theta2_grad += 1/m * Del2;
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
Theta1_grad += (lambda/m)*[zeros(size(Theta1,1),1) regTheta1];
Theta2_grad += (lambda/m)*[zeros(size(Theta2,1),1) regTheta2];
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
%}
function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


a1 = [ones(m,1) X];
a2 = sigmoid(a1*Theta1');
a2 = [ones(m,1) a2];	
a3 = sigmoid(a2*Theta2');
y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);
J = (-1 / m) * sum(sum(y.*log(a3) + (1 - y).*log(1 - a3)));

regTheta1 =  Theta1(:,2:end);
regTheta2 =  Theta2(:,2:end);

error = (lambda/(2*m)) * (sum(sum(regTheta1.^2)) + sum(sum(regTheta2.^2)));

J = J + error;
del1 = zeros(size(Theta1));
del2 = zeros(size(Theta2));
for t = 1:m,
	a1t = a1(t,:);
	a2t = a2(t,:);
	a3t = a3(t,:);
	yt = y(t,:);
	d3 = a3t - yt;
	d2 = Theta2'*d3' .* sigmoidGradient([1;Theta1 * a1t']);
	del1 = del1 + d2(2:end)*a1t;
	del2 = del2 + d3' * a2t;
end;
	% fprintf('\nsize of del1:'); size(del1)
Theta1_grad = 1/m * del1 + (lambda/m)*[zeros(size(Theta1, 1), 1) regTheta1];
Theta2_grad = 1/m * del2 + (lambda/m)*[zeros(size(Theta2, 1), 1) regTheta2];
% fprintf('\nsize of Theta1_grad:'); size(Theta1_grad)
% fprintf('\nsize of Theta2_grad:'); size(Theta2_grad)
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end