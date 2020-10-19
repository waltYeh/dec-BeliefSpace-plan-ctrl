options.rho=1;
options.gam=0.1;
options.reweightedIter=5;
G=sparse_feedback(L_opt,diffs,b_nom,options);
mm = HumanMind(dt); % motion model
om = HumanReactionModel(); % observation model
[didCollide, b0_next, x_true_final] = mpc_centralized_animateGMM(15,16,b0, b_opt, ...
    u_opt, G, update_steps,time_past, mm, om,Op.lims, show_mode);