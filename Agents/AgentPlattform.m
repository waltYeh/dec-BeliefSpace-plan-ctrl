classdef AgentPlattform < AgentBase 
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
        component_stDim = 4;
        %component_stDim + component_stDim^2 + 1
        component_bDim = 21;
        components_amount = 2;
        shared_uDim = 2;
        u_lims = [-0.0 0.0;
            -4.0 4.0;
            -0.0  0.0;
            -0.0  0.0;
            -2.0 2.0;
            -2.0 2.0];
        % larger, less overshoot; smaller, less b-noise affects assist
        P_feedback = 0.3;
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
    end
    
    methods 
        function obj = AgentPlattform(dt_input,horizonSteps)
            obj@AgentBase();       
            obj.dt = dt_input; % delta_t for time discretization
            obj.motionModel = HumanMind(dt_input); % motion model
            obj.obsModel = HumanReactionModel(); % observation model
            obj.dyn_cst  = @(b,u,i) beliefDynCost_assisting_robot(b,u,horizonSteps,false,obj.motionModel,obj.obsModel);
        end
        function [b_nom,u_nom,L_opt,Vx,Vxx,cost]= iLQG_GMM(obj, b0, u_guess, Op)
            [b_nom,u_nom,L_opt,Vx,Vxx,cost,~,~,tt, nIter]= iLQG_GMM(obj.dyn_cst, b0, u_guess, Op);
        end
        function updatePolicy(obj, b_n,u_n,L)
            obj.policyHorizon = size(b_n,2);
            obj.b_nom = b_n;
            obj.u_nom = u_n;
            obj.L_opt = L;
            obj.ctrl_ptr = 1;
        end
        function u = getNextControl(obj, b)
            u = obj.u_nom(:,obj.ctrl_ptr) + obj.P_feedback*obj.L_opt(:,:,obj.ctrl_ptr)*(b - obj.b_nom(:,obj.ctrl_ptr));
            % dim is 6
            for i_u = 1:size(obj.u_nom,1)
                u(i_u)=min(obj.u_lims(i_u,2), max(obj.u_lims(i_u,1), u(i_u)));
            end
            obj.ctrl_ptr = obj.ctrl_ptr + 1;
            if obj.ctrl_ptr > obj.policyHorizon
                error('ctrl_pter out of range');
            end
        end
        function [b_next,mu,sig,weight] = getNextEstimation(obj,b,u,z)
            component_stDim = obj.motionModel.stDim;
            component_bDim = component_stDim + component_stDim^2 + 1;
            shared_uDim = 2;
            component_alone_uDim = obj.motionModel.ctDim - shared_uDim;
            components_amount = length(b)/component_bDim;
            [mu, sig, weight] = b2xPw(b, component_stDim, components_amount);
            z_mu = cell(components_amount);
            z_sig = cell(components_amount);
            for i_comp = 1:components_amount
                %u = [v_ball;v_rest;v_aid_man];
                u_for_comp = [u((i_comp-1)*component_alone_uDim + 1:i_comp*component_alone_uDim);u(end-shared_uDim+1:end)];
                    % Get motion model jacobians and predict pose
            %     zeroProcessNoise = motionModel.generateProcessNoise(mu{i_comp},u_for_comp); % process noise
                zeroProcessNoise = zeros(obj.motionModel.stDim,1);
                x_prd = obj.motionModel.evolve(mu{i_comp},u_for_comp,zeroProcessNoise); % predict robot pose
                A = obj.motionModel.getStateTransitionJacobian(mu{i_comp},u_for_comp,zeroProcessNoise);
                G = obj.motionModel.getProcessNoiseJacobian(mu{i_comp},u_for_comp,zeroProcessNoise);
                Q = obj.motionModel.getProcessNoiseCovariance(mu{i_comp},u_for_comp);
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
                z_ratio = 1;
                if abs(z(1))<1
                    z_ratio = abs(z(1));
                end
                weight_adjust = [z_ratio*weight(i_comp),z_ratio*weight(i_comp),1,1]';
        %         K=weight_adjust.*K;
                P = (eye(obj.motionModel.stDim) - K*H)*P_prd;
                x = x_prd + weight_adjust.*K*(z - z_prd);
                z_mu{i_comp} = z_prd;
                z_sig{i_comp} = HPH;
                mu{i_comp} = x;
                sig{i_comp} = P;
            end

            last_w = weight;

            for i_comp = 1 : components_amount
                weight(i_comp) = last_w(i_comp)*getLikelihood(z - z_mu{i_comp}, z_sig{i_comp} + obj.obsModel.R_w);
            end
            sum_wk=sum(weight);
            if (sum_wk > 0)
                for i_comp = 1 : components_amount
                    weight(i_comp) = weight(i_comp) ./ sum_wk;
                end
            else
                weight = last_w;
            end
            for i_comp = 1 : components_amount
                weight(i_comp) = 0.99*weight(i_comp)+0.01*(1 / components_amount);
            end
            if abs(z(1))<0.25
                weight = last_w;
            end
            b_next = xPw2b(mu, sig, weight, component_stDim, components_amount);
        end
    end
end