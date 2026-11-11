function export_reference(matlab_repo, out_dir)
% Exports reference outputs of the MATLAB fast-partial-tracking code for the
% Python equivalence tests.
%
% matlab_repo: path to the fast-partial-tracking MATLAB repository.
%              Defaults to the FPT_MATLAB_DIR environment variable.
% out_dir:     destination for the .mat files. Defaults to
%              ../tests/fixtures/matlab relative to this script.
%
% Each test case is written to <out_dir>/<case>.mat (MAT v7, readable with
% scipy.io.loadmat). Input signals are generated here and stored in the file
% so both implementations see bit-identical input.
%
% Run headless:
%   matlab -nodisplay -batch "export_reference"

here = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(matlab_repo)
    matlab_repo = getenv('FPT_MATLAB_DIR');
    if isempty(matlab_repo)
        matlab_repo = fullfile(here, '..', '..', 'fast-partial-tracking');
    end
end
if nargin < 2 || isempty(out_dir)
    out_dir = fullfile(here, '..', 'tests', 'fixtures', 'matlab');
end
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end
matlab_repo = canonical(matlab_repo);
out_dir = canonical(out_dir);

addpath(matlab_repo);
addpath(fullfile(matlab_repo, 'utilities'));
addpath(here);

[~, git_hash] = system(sprintf('git -C "%s" rev-parse HEAD', matlab_repo));
git_hash = strtrim(git_hash);
[~, git_dirty] = system(sprintf('git -C "%s" status --porcelain -- "*.m"', matlab_repo));
git_dirty = ~isempty(strtrim(git_dirty));

export_windows(out_dir);
export_assignment(out_dir);

cases = build_cases(matlab_repo);
work_dir = tempname;
mkdir(work_dir);
old_dir = cd(work_dir);
cleanup = onCleanup(@() cd(old_dir));

for c = 1:numel(cases)
    tc = cases(c);
    fprintf('case %s\n', tc.name);
    ref = run_case(tc);
    ref.git_hash = git_hash;
    ref.git_dirty = git_dirty;
    ref.matlab_version = version;
    save(fullfile(out_dir, [tc.name '.mat']), '-struct', 'ref', '-v7');
    if strcmp(tc.name, 'demo')
        copyfile('demo_partials.bin', fullfile(out_dir, 'demo_partials.bin'));
    end
end
fprintf('wrote %d cases to %s\n', numel(cases), out_dir);
end


function p = canonical(p)
old = cd(p);
p = pwd;
cd(old);
end


function cases = build_cases(matlab_repo)
fs = 16000;
t = (0:round(0.5*fs)-1)'/fs;
cases = struct('name', {}, 'x', {}, 'fs', {}, 'params', {});

[x_demo, fs_demo] = audioread(fullfile(matlab_repo, 'audio', 'demo_sound.wav'));
N_demo = 2^nextpow2(.025*fs_demo)-1;
cases(end+1) = mk('demo', x_demo(:,1), fs_demo, [N_demo 4 2 -50 2 .2 50 15]);

x = 0.5*sin(2*pi*440*t + 0.3);
cases(end+1) = mk('sine', x, fs, [1023 4 1 -40 2 .2 50 15]);

% Harmonic exponential chirp 200 -> 2000 Hz.
f0 = 200*(10.^(t/t(end)));
ph = 2*pi*cumsum(f0)/fs;
x = zeros(size(t));
for h = 1:5
    x = x + (0.4/h)*sin(h*ph);
end
cases(end+1) = mk('chirp', x, fs, [1023 4 2 -60 2 .2 50 15]);
cases(end+1) = mk('chirp_params', x, fs, [1023 4 2 -60 2 .5 100 10]);

% Harmonic tone with vibrato (+-1 semitone at 6 Hz), even window length.
f0 = 330*2.^(sin(2*pi*6*t)/12);
ph = 2*pi*cumsum(f0)/fs;
x = zeros(size(t));
for h = 1:8
    x = x + (0.3/h)*cos(h*ph + h);
end
cases(end+1) = mk('vibrato', x, fs, [1024 4 1 -60 3 .2 50 15]);

% Two crossing linear chirps.
f1 = 500 + 2000*t;
f2 = 1500 - 2000*t;
x = 0.4*sin(2*pi*cumsum(f1)/fs) + 0.3*sin(2*pi*cumsum(f2)/fs + 1);
cases(end+1) = mk('crossing', x, fs, [1024 4 2 -60 1 .2 50 15]);

% Gated tone bursts (births and deaths) in low-level noise.
rng(1);
g1 = double(t > 0.05 & t < 0.20);
g2 = double(t > 0.15 & t < 0.40);
g3 = double(t > 0.30 & t < 0.45);
x = g1.*0.5.*sin(2*pi*600*t) + g2.*0.3.*sin(2*pi*1250*t + 0.5) ...
    + g3.*0.2.*sin(2*pi*3100*t + 2) + 1e-3*randn(size(t));
cases(end+1) = mk('bursts', x, fs, [511 4 1 -50 2 .2 50 15]);

rng(0);
x = 0.1*randn(size(t));
cases(end+1) = mk('noise', x, fs, [1023 4 1 -30 2 .2 50 15]);
end


function tc = mk(name, x, fs, params)
tc = struct('name', name, 'x', x, 'fs', fs, 'params', params);
end


function ref = run_case(tc)
pc = num2cell(tc.params);
[Partials_mem, time_mem, padding, L, S] = jun_track_partials(tc.x, tc.fs, pc{:}); %#ok<ASGLU>
[Partials, time, padding_f, L_f, fs_f, num_tracks] = jun_read_partials('demo_partials.bin');

s = fpt_setup(tc.x, tc.fs, pc{:});

rows = zeros(0, 7);
for m = 1:numel(Partials)
    P = Partials{m};
    if ~isempty(P)
        r = size(P, 1);
        rows = [rows; repmat([m time(m)], r, 1) P(:,1:4) (1:r)']; %#ok<AGROW>
    end
end

yfull = jun_synthesize_partials(Partials, time, L_f + sum(padding_f));
y = yfull(padding_f(1)+1:end-padding_f(2));

ref = struct();
ref.name = tc.name;
ref.x = tc.x;
ref.fs = tc.fs;
ref.params = tc.params;
ref.N = s.N; ref.H = s.H; ref.Ndft = s.Ndft; ref.M = s.M; ref.L = s.L;
ref.hop_factor = s.HopFactor; ref.oversample = s.OverSample; ref.Q = s.Q;
ref.n_center = s.n_center; ref.padding = s.padding;
ref.threshold = s.G_g; ref.maxS = s.maxS; ref.max_sample = s.max_sample;
ref.varF = s.varF; ref.varA = s.varA; ref.zetaA = s.zetaA;
ref.win = s.win; ref.winD = s.winD; ref.n_midpoint = s.n_midpoint;
ref.time = time;
ref.num_tracks = num_tracks;
% rows: frame (1-based), time (1-based padded sample), omega, log-amp, phase, track (1-based), row
ref.partials = rows;
ref.yfull = yfull;
ref.y = y;
ref.rsnr = 10*log10(sum(tc.x.^2)/sum((tc.x - y).^2));

% Per-frame DDM intermediates for a few frames.
frames = unique(max(2, min(s.M, [2 round(s.M/3) round(s.M/2) round(2*s.M/3) s.M-1])));
ref.ddm_frames = frames;
ref.ddm = cell(numel(frames), 1);
ref.cost = cell(numel(frames), 1);
for f = 1:numel(frames)
    m = frames(f);
    [A, npk, Sm, pk] = ddm_frame(s, m);
    [A_last, npk_last] = ddm_frame(s, m-1);
    ref.ddm{f} = struct('frame', m, 'alpha', A, 'num_peaks', npk, 'S', Sm, ...
        'peak_bin', pk.peak, 'left_bin', pk.left, 'right_bin', pk.right);
    ref.cost{f} = cost_frame(s, A_last, npk_last, A, npk);
end
end


function [Alpha, num_peaks, S, pk] = ddm_frame(s, m)
time_m = 1 + (m-1)*s.H + s.n_center;
yst = s.ypad(time_m-s.n_center:time_m+s.N-1-s.n_center);
[Alpha, num_peaks, S] = jun_ddm(yst, s.Q, s.Rddm, s.G_g, s.win, s.winD, ...
    s.p, s.pD, s.centering, s.Ndft, s.ft_mat, s.omega);
[peak, ~, left, right] = jun_pick_peaks(S(1:s.Ndft/2), s.G_g);
pk = struct('peak', peak, 'left', left, 'right', right);
if num_peaks == 0
    Alpha = zeros(s.Q+1, 0);
end
end


function c = cost_frame(s, Alpha_last, num_peaks_last, Alpha, num_peaks)
c = struct('C', zeros(num_peaks_last, num_peaks), 'type', zeros(num_peaks_last, num_peaks), ...
    'assign', zeros(num_peaks_last, 1), 'useful_rows', zeros(0,1), 'useful_cols', zeros(0,1));
if num_peaks == 0 || num_peaks_last == 0
    return
end
p = s.p; pD = s.pD; nm = s.n_midpoint; fs = s.fs;
mA1 = real(p(nm(2),:)*Alpha_last);
mA2 = real(p(nm(1),:)*Alpha);
mF1 = fs/(2*pi)*imag(pD(nm(2),:)*Alpha_last(2:end,:));
mF2 = fs/(2*pi)*imag(pD(nm(1),:)*Alpha(2:end,:));
DeltaA = bsxfun(@minus, mA1', mA2);
DeltaF = bsxfun(@minus, mF1', mF2);
A_useful = 1-exp(-DeltaF.^2/s.varF - DeltaA.^2/s.varA);
B_spurious = 1-(1-s.delta)*A_useful;
[Costs, Type] = min([B_spurious(:) A_useful(:)],[],2);
C_mtx = reshape(abs(Costs), num_peaks_last, num_peaks);
jj = munkres(C_mtx);
ii = find(jj);
ind = ii + num_peaks_last*(jj(ii)-1);
ii = ii(Type(ind)==2);
c.C = C_mtx;
c.type = reshape(Type, num_peaks_last, num_peaks);
c.assign = jj(:);
c.useful_rows = ii(:);
c.useful_cols = reshape(jj(ii), [], 1);
c.mA1 = mA1; c.mA2 = mA2; c.mF1 = mF1; c.mF2 = mF2;
end


function export_windows(out_dir)
lengths = [7 8 511 1023 1024 2047];
ref = struct();
for k = 1:numel(lengths)
    N = lengths(k);
    ref.(sprintf('hann_%d', N)) = [jun_hannwindow(N,1) jun_hannwindow(N,2)];
    ref.(sprintf('bh_%d', N)) = [jun_bhwindow(N,1) jun_bhwindow(N,2)];
end
ref.lengths = lengths;
save(fullfile(out_dir, 'windows.mat'), '-struct', 'ref', '-v7');
end


function export_assignment(out_dir)
rng(3);
shapes = [5 5; 3 7; 9 4; 1 6; 12 12; 30 25];
cases = cell(size(shapes,1), 1);
for k = 1:size(shapes,1)
    C = rand(shapes(k,1), shapes(k,2));
    [jj, cost] = munkres(C);
    cases{k} = struct('C', C, 'assign', jj(:), 'cost', cost);
end
ref.cases = cases;
save(fullfile(out_dir, 'assignment.mat'), '-struct', 'ref', '-v7');
end
