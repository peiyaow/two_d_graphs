import cvxpy as *
obj = 0
# sum_D = 0
for i in range(n):
    S = semidefinite(p, name='S')
#    D = Variable(p,p, name='D')
#    D = S-Omega
    obj = obj - log_det(S) + trace(C_MDD_array[i] * S) + lamb*norm(S,1)
#    sum_D = sum_D + D
objective = Minimize(obj)
# constraints = [sum_D == 0]
# prob = Problem(objective, constraints)
prob = Problem(objective)
prob.solve()

#####
gvx = TGraphVX()
C_MDD_array = C_MDD_array[range(11)+range(12,14)]
n = C_MDD_array.shape[0]
indexOfPenalty = 3
lamb = alphas[5] # D
beta = alphas[5] # Omega
for i in range(n):
    n_id = i
    S = semidefinite(p, name='S')
#    D = Variable(p,p, name='D')
    obj = -log_det(S) + trace(C_MDD_array[i] * S)  # + alpha*norm(S,1)
    gvx.AddNode(n_id, obj)

    # if i > 0:  # Add edge to previous timestamp
    #     prev_Nid = n_id - 1
    #     currVar = gvx.GetNodeVariables(n_id)
    #     prevVar = gvx.GetNodeVariables(prev_Nid)
    #     currScore = score[n_id]
    #     prevScore = score[prev_Nid]
    #     edge_obj = beta/np.abs(currScore - prevScore) * norm(currVar['S'] - prevVar['S'], indexOfPenalty)
    #     gvx.AddEdge(n_id, prev_Nid, Objective=edge_obj)

    # Add fake nodes, edges
    gvx.AddNode(n_id + n)
    gvx.AddEdge(n_id, n_id + n, Objective=beta * norm(S, 1))
    #lamb*norm(S-Omega, 1)
    # gvx.AddNode(n_id + 2*n)
    # gvx.AddEdge(n_id, n_id + 2*n, Objective=beta * norm(S, 1))

verbose = True
epsAbs = 1e-3
epsRel = 1e-3
eps = 3e-3

gvx.Solve(EpsAbs=epsAbs, EpsRel=epsRel, Verbose=verbose)

thetaSet = []
for nodeID in range(n):
    val = gvx.GetNodeValue(nodeID, 'S')
    thetaEst = upper2FullTVGL(val, eps)
    thetaSet.append(thetaEst)

G1 = nx.from_numpy_array(thetaSet[3]-Omega)
nx.draw(G1)

alpha_i = [alpha_max(C_MDD_array[i]) for i in range(n)]

C_MDD_array = C_MDD_array[range(1)+range(2,n)]
alpha_max(C_MDD_array[11])
