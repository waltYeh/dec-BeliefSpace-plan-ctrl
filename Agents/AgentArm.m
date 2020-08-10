classdef AgentArm < AgentBase 
    properties (Constant = true) 
        % note that all properties must be constant, because we have a lot of copies of this object and it can take a lot of memory otherwise.
%         compStateDim = 4; % state dimension
%         ctDim = 6;  % control vector dimension
%         bDim = 42;
%         wDim = 4;   % Process noise (W) dimension
%         P_Wg; % covariance of state-additive-noise
%         sigma_b_u; % A constant bias intensity (covariance) of the control noise
%         eta_u; % A coefficient, which makes the control noise intensity proportional to the control signal       
%         zeroNoise;
%         ctrlLim; % control limits
        component_stDim = 2;
        %component_stDim + component_stDim^2 + 1
        component_bDim = 6;
        components_amount = 1;
        shared_uDim = 0;
        total_uDim = 2;
        u_lims = [-4.0 4.0;
            -4.0 4.0];
        % larger, less overshoot; smaller, less b-noise affects assist
        
    end
    
    properties
%         dt = 0.05; % delta_t for time discretization
%         motionModel = HumanMind(dt); % motion model
%         obsModel = HumanReactionModel(); % observation model
        policyHorizon;
        b_nom;
        u_nom;
        L_opt;
        ctrl_ptr;
        digraph_idx;
        derivatives;
        lambda; 
        dlambda;
        flgChange;
        P_feedback = 1.0;
    end
    
    methods 
        function obj = AgentArm(dt_input,horizonSteps, node_idx, belief_dyns)
            obj@AgentBase();  
            obj.digraph_idx = node_idx;
            obj.derivatives = {};
            obj.dt = dt_input; % delta_t for time discretization
            obj.motionModel = TwoDPointRobot(dt_input); % motion model
            obj.obsModel = TwoDSimpleObsModel(); % observation model
            obj.dyn_cst  = @(D,idx,b,u,i) beliefDynCost_crane(D,idx,b,u,horizonSteps,false,obj.motionModel,obj.obsModel, belief_dyns);
            obj.lambda = []; 
            obj.dlambda = [];
            obj.flgChange = [];
        end
        
        function [b_nom,u_nom,L_opt,Vx,Vxx,cost]= iLQG_agent(obj,D, b0, u_guess, Op)
            [b_nom,u_nom,L_opt,Vx,Vxx,cost,~,~,tt, nIter]= iLQG_multiagent(D,obj.digraph_idx,obj.dyn_cst, b0, u_guess, Op);
        end
        function [x, u, cost, L, Vx, Vxx, finished]= ...
                iLQG_one_it...
                (obj,D, b0, Op, iter, u_guess, u_last,x_last, cost_last)
            if iter == 1
                obj.lambda = [];
                obj.dlambda = [];
                obj.flgChange = [];
            end
            [x, u, L, Vx, Vxx, cost, obj.lambda, obj.dlambda, finished,obj.flgChange,derivatives_cell] = ...
                iLQG_hetero_multiagent_one_iter(D,obj.digraph_idx,obj.dyn_cst, b0, Op, iter,...
                u_guess,obj.lambda, obj.dlambda, u_last,x_last, cost_last,obj.flgChange,obj.derivatives, obj.u_lims);
            obj.derivatives = derivatives_cell;
        end
        function updatePolicy(obj,b_n_idx,u_n_idx,L)
            %b_n 6x61, u_n 2x60, L 2x6x60
            obj.policyHorizon = size(b_n_idx{obj.digraph_idx},2);
            obj.b_nom = b_n_idx;
            obj.u_nom = u_n_idx;
            obj.L_opt = L;
            obj.ctrl_ptr = 1;
        end
        function u = getNextControl(obj, b)
            diff_b = b{obj.digraph_idx} - obj.b_nom{obj.digraph_idx}(:,obj.ctrl_ptr);
            u = obj.u_nom{obj.digraph_idx}(:,obj.ctrl_ptr) + obj.P_feedback*obj.L_opt(:,:,obj.ctrl_ptr)*diff_b;
            % dim is 6
            for i_u = 1:size(obj.u_lims,1)
                u(i_u)=min(obj.u_lims(i_u,2), max(obj.u_lims(i_u,1), u(i_u)));
            end
            obj.ctrl_ptr = obj.ctrl_ptr + 1;
            if obj.ctrl_ptr > obj.policyHorizon
                error('ctrl_pter out of range');
            end
        end
        function [b_next,mu,sig,weight] = getNextEstimation(obj,b,u,z)
            component_alone_uDim = obj.motionModel.ctDim - obj.shared_uDim;
%             components_amount = length(b)/component_bDim;
%             [mu, sig, weight] = b2xPw(b, obj.component_stDim, obj.components_amount);
            mu=cell(1);
            sig=cell(1);
            [mu{1}, sig{1}] = b2xP(b, obj.component_stDim);
            z_mu = cell(obj.components_amount);
            z_sig = cell(obj.components_amount);
            for i_comp = 1:obj.components_amount
                %u = [v_ball;v_rest;v_aid_man];
                u_for_comp = [u((i_comp-1)*component_alone_uDim + 1:i_comp*component_alone_uDim);u(end-obj.shared_uDim+1:end)];
                    % Get motion model jacobians and predict pose
            %     zeroProcessNoise = motionModel.generateProcessNoise(mu{i_comp},u_for_comp); % process noise
                zeroProcessNoise = zeros(obj.motionModel.stDim,1);
                x_prd = obj.motionModel.evolve(mu{i_comp},u_for_comp,zeroProcessNoise); % predict robot pose
                A = obj.motionModel.getStateTransitionJacobian(mu{i_comp},u_for_comp,zeroProcessNoise);
                G = obj.motionModel.getProcessNoiseJacobian(mu{i_comp},u_for_comp,zeroProcessNoise);
                Q = obj.motionModel.Q_est;%getProcessNoiseCovariance(mu{i_comp},u_for_comp);
                P_prd = A*sig{i_comp}*A' + G*Q*G';

                z_prd = obj.obsModel.getObservation(x_prd,'nonoise'); % predicted observation
                zerObsNoise = zeros(length(z),1);
                H = obj.obsModel.getObservationJacobian(mu{i_comp},zerObsNoise);
                % M is eye
                M = obj.obsModel.getObservationNoiseJacobian(mu{i_comp},zerObsNoise,z);
            %     R = obsModel.getObservationNoiseCovariance(x,z);
            %     R = obsModel.R_est;
                % update P
                HPH = H*P_prd*H';
            %     S = H*P_prd*H' + M*R*M';
                K = (P_prd*H')/(HPH + M*obj.obsModel.R_est*M');
%                 z_ratio = 1;
%                 if abs(z(1))<1
%                     z_ratio = abs(z(1));
%                 end
%                 weight_adjust = [z_ratio*weight(i_comp),z_ratio*weight(i_comp),1,1]';
        %         K=weight_adjust.*K;
                P = (eye(obj.motionModel.stDim) - K*H)*P_prd;
                x = x_prd + K*(z - z_prd);
                z_mu{i_comp} = z_prd;
                z_sig{i_comp} = HPH;
                mu{i_comp} = x;
                sig{i_comp} = P;
            end
            if obj.components_amount> 1
                last_w = weight;

                for i_comp = 1 : obj.components_amount
                    weight(i_comp) = last_w(i_comp)*getLikelihood(z - z_mu{i_comp}, z_sig{i_comp} + obj.obsModel.R_w);
                end
                sum_wk=sum(weight);
                if (sum_wk > 0)
                    for i_comp = 1 : obj.components_amount
                        weight(i_comp) = weight(i_comp) ./ sum_wk;
                    end
                else
                    weight = last_w;
                end
                for i_comp = 1 : obj.components_amount
                    weight(i_comp) = 0.99*weight(i_comp)+0.01*(1 / obj.components_amount);
                end
                if abs(z(1))<0.25
                    weight = last_w;
                end
                b_next = xPw2b(mu, sig, weight, obj.component_stDim, obj.components_amount);
            else
                b_next = [mu{1};sig{1}(:)];
                weight = [];
            end
            
        end
    end
end
