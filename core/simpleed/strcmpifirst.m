function r = strcmpifirst(s1, s2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% 22-Jan-2015

l = min(length(s1), length(s2));
r = strcmpi(s1(1:l), s2(1:l));

end

