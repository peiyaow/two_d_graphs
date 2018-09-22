clear;
clc;
load('var_Y_array.mat')
%Sd = cell(1, n_blocks);
n_blocks = 50;
n = 116;

for i=1:n_blocks
    %R    = randn(n, m(i));
    Sd{i} = squeeze(var_Y_array(i,:,:));
end

%cvx_solver mosek;
cvx_begin sdp;
    variable S(n,n) symmetric;
    fx = 0;
    for i=1:n_blocks
        fx = fx + vec(S - Sd{i})'*vec(S - Sd{i});
    end
    
    minimize( fx );
    subject to 
        for i=1:n_blocks
            Sd{i} - S >= 0;
        end
        S >= 0;
cvx_end