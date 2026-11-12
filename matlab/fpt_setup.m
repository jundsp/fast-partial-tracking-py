function s = fpt_setup(y, fs, N, HopFactor, OverSample, G_g, Q, delta, zetaF, zetaA)
% Reproduces the SETUP section of jun_track_partials.m verbatim so that
% export_reference.m can call jun_ddm on individual frames and capture
% intermediate results. Keep this in sync with jun_track_partials.m.

N = max(4,round(N));
HopFactor = 2^nextpow2(max(1,HopFactor));
OverSample = 2^nextpow2(max(1,OverSample));
G_g = abs(G_g);
Q = max(1,round(Q));
zetaF(zetaF<1e-1) = 1e-1;
zetaA(zetaA<1e-1) = 1e-1;
zetaA = zetaA/20*log(10);

y = y(:,1);
L = length(y);

varF = -zetaF^2/log((delta-1)/(delta-2));
varA = -zetaA^2/log((delta-1)/(delta-2));

if OverSample > 1
    win = jun_bhwindow(N,1);
    winD = jun_bhwindow(N,2);
    HopFactor = max(8,HopFactor);
else
    win = jun_hannwindow(N,1);
    winD = jun_hannwindow(N,2);
    HopFactor = max(4,HopFactor);
end

H = round((N-mod(N,2))/HopFactor);
Ndft = 2^(round(log2(OverSample*2^round(log2(N)))));
if Ndft < N
    Ndft = 2^round(log2(N)+1);
end
ndft = (0:Ndft-1)';

padding = zeros(2,1);
padding(1) = (Ndft-H);
M = ceil((Ndft+L - 2*H-1)/H + 1);
padding(2) = ((M-1)*H + Ndft) - (padding(1) + L);

ypad = [zeros(padding(1),1); y; zeros(padding(2),1)];

omega = 2*pi*ndft/Ndft;
n_center = (N - mod(N,2))/2+1;
ndft = ndft - n_center;
centering = exp(1j*omega*n_center);
ft_mat = exp(1j*ndft*omega(1:Ndft/2)');
Rddm = Q*OverSample;

n = (0:N-1)' - n_center;
i = (0:Q);
p = bsxfun(@power, n, i);
pD = bsxfun(@times, bsxfun(@power, n, (0:Q-1)), 1:Q);
n_midpoint = [-floor(H/2) ceil(H/2)] + n_center + 1;

[~, max_sample] = max(abs(ypad));
maxS = 20*log10(max(abs(fft(win.*ypad(max_sample + (0:N-1)-n_center),Ndft))));
G_g = maxS - G_g;

s = struct('N',N,'HopFactor',HopFactor,'OverSample',OverSample,'G_g',G_g, ...
    'maxS',maxS,'Q',Q,'delta',delta,'zetaF',zetaF,'zetaA',zetaA,'varF',varF, ...
    'varA',varA,'win',win,'winD',winD,'H',H,'Ndft',Ndft,'padding',padding, ...
    'M',M,'L',L,'ypad',ypad,'omega',omega,'n_center',n_center, ...
    'centering',centering,'ft_mat',ft_mat,'Rddm',Rddm,'p',p,'pD',pD, ...
    'n_midpoint',n_midpoint,'max_sample',max_sample,'fs',fs);
end
