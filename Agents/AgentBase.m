classdef AgentBase < handle
    properties (Abstract, Constant) 
        % note that all properties must be constant, because we have a lot of copies of this object and it can take a lot of memory otherwise.
%         stDim; % state dimension
%         ctDim;  % control vector dimension
%         bDim;
%         wDim;   % Process noise (W) dimension
%         P_Wg; % covariance of state-additive-noise
%         sigma_b_u; % A constant bias intensity (covariance) of the control noise
%         eta_u; % A coefficient, which makes the control noise intensity proportional to the control signal       
%         zeroNoise;
%         ctrlLim; % control limits
        u_lims;
        P_feedback;
    end
    
    properties
        dt; % delta_t for time discretization
%         horizonSteps;
        motionModel; % motion model
        obsModel; % observation model
        dyn_cst;
    end
    
    methods (Abstract)
        
    end
end