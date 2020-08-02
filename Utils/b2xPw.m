function [x, P, w] = b2xPw(b, component_stDim, components_amount)
component_bDim = component_stDim + component_stDim^2 + 1;
x = cell(components_amount,1);
P = cell(components_amount,1);
w = zeros(components_amount,1);
% for para = 1:size(b,2)
    for i_comp=1:components_amount
        b_comp = b((i_comp-1)*component_bDim+1:i_comp*component_bDim);
        x{i_comp} = b_comp(1:component_stDim);
        for d = 1:component_stDim
            P{i_comp}(:,d) = b_comp(d*component_stDim+1:(d+1)*component_stDim, 1);
        end
        w(i_comp) = b_comp(end);
    end
% end
end

