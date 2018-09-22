
n = 10;
m = [5, 15, 3];
n_blocks = 3;
%Sd = cell(1, n_blocks);
for i=1:n_blocks
    R    = randn(n, m(i));
    Sd{1,i} = R*R';
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


