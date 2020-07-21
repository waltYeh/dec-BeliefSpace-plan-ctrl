function b = xPw2b(x, P, w, component_stDim, components_amount)
component_bDim = component_stDim + component_stDim^2 + 1;
b = zeros(component_bDim*components_amount,1); % current belief
for i_comp=1:components_amount
    b((i_comp-1)*component_bDim+1:(i_comp-1)*component_bDim+component_stDim)=x{i_comp};
    b((i_comp-1)*component_bDim+component_stDim+1:(i_comp-1)*component_bDim+component_stDim+component_stDim*component_stDim)=P{i_comp}(:);
    b((i_comp)*component_bDim)=w(i_comp);
end
end

