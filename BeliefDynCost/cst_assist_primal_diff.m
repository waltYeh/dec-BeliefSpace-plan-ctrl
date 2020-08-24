function [c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj] ...
        = cst_assist_primal_diff(D,idx,b,u,c_bi,c_ui,...
        c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj,lam_di,lam_up,rho_d,rho_up)
horizon = size(c_bi,2);
belief_dim = size(c_bi,1);
ctrl_dim = size(c_ui,1);
stDim=2;
n_agent = size(D.Nodes,1);
for k=1:horizon-1
%     c_bi(:,k) = c_bi(:,k) + rho(1) * (x{idx}(:,k) - );
    uj_sum = zeros(ctrl_dim,1);
    for j = 2:n_agent
        uj_sum = uj_sum + u{j}(:,k);
    end   
    c_ui(:,k) = c_ui(:,k) -rho_up * (3*u{1}(5:6,k) - uj_sum + transpose(lam_up(1,:,k)));
    c_ui_ui(:,:,k) = c_ui_ui(:,:,k) + rho_up * eye(ctrl_dim);
end
for k=1:horizon
    components_amount=2;
    stDim_platf = 4;
    [x_platf_comp, P_platf, w] = b2xPw(b{1}(:,k), stDim_platf, components_amount);

    x_platf_weighted = zeros(2,components_amount);
    for i=1:components_amount
        x_platf_weighted(:,i)=transpose(x_platf_comp{i}(3:4)*w(i));
    end
    x_platf= [sum(x_platf_weighted(1,:));sum(x_platf_weighted(2,:))];
    edge_row = idx-1;
    formation_residue = b{idx}(1:stDim,k)-x_platf-(D.Edges.nom_formation_2(edge_row,:))';
    x_in_b = 1:2;
    c_bi(x_in_b,k) = c_bi(x_in_b,k) ...
                + rho_d * (formation_residue + transpose(lam_di(idx-1,:,k)));
    c_bi_bi(x_in_b,x_in_b,k) = c_bi_bi(x_in_b,x_in_b,k) ...
                + rho_d * eye(stDim);
end
end