function [c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj] ...
        = cst_plattform_primal_diff(D,idx,b,u,c_bi,c_ui,...
        c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj,lam_di,lam_up,rho_d,rho_up)
horizon = size(c_bi,2);
belief_dim = size(c_bi,1);
comp_amount = 2;
single_comp_dim = belief_dim / comp_amount;
ctrl_dim = 2;
stDim=2;
n_agent = size(D.Nodes,1);
for k=1:horizon-1
%     c_bi(:,k) = c_bi(:,k) + rho(1) * (x{idx}(:,k) - );
    uj_sum = zeros(ctrl_dim,1);
    for j = 2:n_agent
        uj_sum = uj_sum + u{j}(:,k);
    end   
    c_ui(5:6,k) = c_ui(5:6,k) + ...
        rho_up * 3*(3*u{idx}(5:6,k) - uj_sum + transpose(lam_up(1,:,k)));
    c_ui_ui(5:6,5:6,k) = c_ui_ui(5:6,5:6,k) + ...
        rho_up * 3 * 3 * eye(ctrl_dim);
end
[eid,nid] = inedges(D,idx);
for i_comp = 1:1
    for k=1:horizon
        for j_nid = 1:length(nid)
            j = nid(j_nid);
            edge_row = eid(j_nid);
            
            xj = b{j}(1:stDim,1);
            x_in_b = (i_comp-1)*single_comp_dim+3:(i_comp-1)*single_comp_dim+4;
            x_Plattform = b{idx}(x_in_b,k);
            formation_residue = xj-x_Plattform-(D.Edges.nom_formation_2(edge_row,:))';
            c_bi(x_in_b,k) = c_bi(x_in_b,k) ...
                - rho_d * (formation_residue + transpose(lam_di(j-1,:,k)));
            c_bi_bi(x_in_b,x_in_b,k) = c_bi_bi(x_in_b,x_in_b,k) ...
                + rho_d * eye(stDim);
        end

    end
end
end