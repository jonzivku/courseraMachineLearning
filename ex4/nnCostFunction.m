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

% turn y values into matrix
%what is this function doing?
y_matrix = eye(num_labels)(y,:); % 5000x10 
                                 % size(y_matrix)

% add bias element to X, make a1

a1 = [ones(m, 1) X]; % Size 5000 x 401

% calculate a2, add bias element to a2

a2 = [ones(m,1) sigmoid(a1*Theta1.')]; %size 5000 26

%calculate a3
a3 = sigmoid(a2*Theta2.'); %size 5000 x 10 , this is also htheta(x)

%calculate J(without regularization)
J = (-1/m).*sum( sum( y_matrix.*log(a3) + (1 - y_matrix).*log(1-a3), 2)) ;

%calculate J(with regularization) 
%check out this column reduction Theta1(:, 2:end)
sT1 = sum( sum(Theta1(:,2:end).^2, 2) );
sT2 = sum( sum(Theta2(:,2:end).^2, 2) );
J = J + (lambda/(2*m))*(sT1 + sT2);

    
%d3 = zeros(size(y_matrix));
d3 = a3 - y_matrix;

z2 = a1 * Theta1.';

d2 = d3 * Theta2(:,2:end) .* sigmoidGradient(z2);

Delta1 = d2.' * a1;

Delta2 = d3.' * a2;

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

Theta1(:,1) = 0;
Theta2(:,1) = 0;

Theta1_grad = Theta1_grad + (lambda*Theta1)/m;
Theta2_grad = Theta2_grad + (lambda*Theta2)/m;



#{
for i = 1:1 %change this to m
    a1 = X(i,:); % 1 400
    a2 = sigmoid([1 a1]*Theta1.'); % (1 x 401) x ( 25 x 401)'
                                   % = (1 x 25) 
    a3 = sigmoid([1 a2]*Theta2.'); % 1 x 10
    delta3 = zeros(
    delta3 = a3 - y(i, :); % 1 x 10
                           %  delta2 = delta3'*Theta2.*(sigmoidGradient(a2*Theta2)).';
    delta2 = delta2(2:end);

end
#}






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
