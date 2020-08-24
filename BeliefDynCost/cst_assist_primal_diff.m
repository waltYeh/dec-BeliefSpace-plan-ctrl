function [c_bi,c_ui,c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj] ...
        = cst_assist_primal_diff(D,idx,x,u,c_bi,c_ui,...
        c_bi_bi,c_bi_ui,c_ui_ui,c_ui_uj,lam_di,lam_up,rho_d,rho_up)
horizon = size(c_bi,2);
belief_dim = size(c_bi,1);
ctrl_dim = size(c_ui,1);
c_ui_inc = c_ui;
c_ui_ui_inc = c_ui_ui;
for k=1:horizon
%     c_bi(:,k) = c_bi(:,k) + rho(1) * (x{idx}(:,k) - );
    c_ui_inc(:,k) = rho(2) * (u{idx}(:,k) - uC_lambda{idx}(:,k));
    c_ui_ui_inc(:,:,k) = rho(2) * eye(ctrl_dim);
    
    c_ui(:,k) = c_ui(:,k) + c_ui_inc(:,k);
    c_ui_ui(:,:,k) = c_ui_ui(:,:,k) + c_ui_ui_inc(:,:,k);
end
a=1;
end