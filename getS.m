clear;
clc;
load('var_Y_array_Shuheng.mat')
%Sd = cell(1, n_blocks);

n = 116;
time_ix = 1:30:180;
n_blocks = length(time_ix);
for i=1:n_blocks
    %R    = randn(n, m(i));
    ix_Y = time_ix(i);
    Sd{i} = squeeze(var_Y_array(ix_Y,:,:));
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

%cvx_solver mosek;
cvx_begin sdp;
    variable S(n,n) symmetric;
    fx = 0;
    for i=1:n_blocks
        fx = fx + max(sum(abs(Sd{i}-S),2));
    end
    
    minimize( fx );
    subject to 
        for i=1:n_blocks
            Sd{i} - S >= 0;
        end
        S >= 0;
cvx_end

