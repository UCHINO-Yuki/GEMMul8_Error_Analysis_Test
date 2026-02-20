function plot_err(k)

arguments (Input)
    k (1,1) double = 8192
end

gd = gpuDevice();
m = 128;
n = 128;
N_d = 3:20;
N_s = 3:15;
PHI_d = 0:0.5:4;
PHI_s = 0:0.5:4;

dd.numSplit(5);
u64 = dd(2^-53);
v = dd(ones(k,1));
for phi = PHI_d
    rng(1,"twister");
    hA = (rand(m,k)-0.5) .* exp(randn(m,k).*phi);
    hB = (rand(k,n)-0.5) .* exp(randn(k,n).*phi);
    A = gpuArray( hA );
    B = gpuArray( hB );
    wait(gd); C = dd(A)*dd(B); wait(gd);
    C = dd(gather(C));
    alpha = 2.^( floor( log2( abs( max(dd(abs(hA)),[],2) ) ) ) );
    beta  = 2.^( floor( log2( abs( max(dd(abs(hB)),[],1) ) ) ) );
    err_max = zeros(length(N_d),1);
    err_min = zeros(length(N_d),1);
    est_max2 = zeros(length(N_d),1);
    est_min2 = zeros(length(N_d),1);
    est_max1 = zeros(length(N_d),1);
    est_min1 = zeros(length(N_d),1);
    for i = 1:length(N_d)
        fprintf('(phi, N) = (%.1f,%d)\n', phi, N_d(i));

        wait(gd); D = oz2(A,B,N_d(i),0); wait(gd);
        err = abs(double(gather(D)-C));
        [err_min(i), err_max(i)] = bounds(err,'all');

        p = mutually_coprime(256,N_d(i));
        [e,f,Cint,Cint2] = calc_Cmax(hA,hB,p);
        P = dd(1);
        for ip=1:length(p)
            P = P * dd(p(ip));
        end

        t = sqrt(2^(-5)./(P-1));
        rho = sum(dd(floor(p./2)));
        r64 = (1 + 3*u64) * 2^(1+ceil(log2(rho))) * (dd(N_d(i)) + 2) * u64^2 * rho * P + 3*u64/2 * P;
        % r64 = (1 + 3*u64) * 2^(1+ceil(log2(rho))) * (dd(N_d(i)) + 2) * u64^2 * rho * P + 3*u64*Cint2;
        est2 = t*(abs(hA)*v)*(beta .* sqrt(f)) ...
            + t*(alpha .* sqrt(e))*(v'*abs(hB)) ...
            + (k + r64).*(t^2*(alpha .* sqrt(e))*(beta .* sqrt(f)));
        [est_min2(i), est_max2(i)] = bounds(double(est2),'all');

        R64 = (1 + 3*u64) * 2^(1+ceil(log2(rho))) * (dd(N_d(i)) + 2) * u64^2 * rho * P + 3*u64*abs(Cint);
        est1 = t*(abs(hA)*v)*(beta .* sqrt(f)) ...
            + t*(alpha .* sqrt(e))*(v'*abs(hB)) ...
            + (k + R64).*(t^2*(alpha .* sqrt(e))*(beta .* sqrt(f)));
        [est_min1(i), est_max1(i)] = bounds(double(est1),'all');
    end

    C64 = A*B;
    err_64f = abs(double(gather(D)-C64));
    [err_64f_min, err_64f_max] = bounds(err_64f,'all');

    fig = figure; 
    plot(N_d,est_max1,'-r', ...
        N_d,est_min1,':r', ...
        N_d,est_max2,'-g', ...
        N_d,est_min2,':g', ...
        N_d,err_max,'-b', ...
        N_d,err_min,':b', ...
        N_d,err_64f_max*ones(size(N_d)),'--k', ...
        'LineWidth',1.5);
    set(gca,'YScale','Log','FontSize',14);
    xlim([min(N_d), max(N_d)]);
    xticks(N_d)
    grid on;
    xlabel('Number of moduli');
    ylabel('Absolute error')
    title("$\phi = " + phi + "$",'Interpreter','latex');
    legend("est\_max","est\_min", ...
        "est2\_max","est2\_min", ...
        "err\_max","err\_min", ...
        "err64\_max", ...
        'Location','best');
    p = char(string(phi*10 + 100));
    savefig(fig, "fig/d_phi=" + p(2:3));
    exportgraphics(fig,"fig/d_phi=" + p(2:3) + ".png",'Resolution',600);
end

u32 = dd(2^-24);
for phi = PHI_s
    rng(1,"twister");
    hA = double( single( (rand(m,k)-0.5) .* exp(randn(m,k).*phi) ) );
    hB = double( single( (rand(k,n)-0.5) .* exp(randn(k,n).*phi) ) );
    A = gpuArray( single( hA ) );
    B = gpuArray( single( hB ) );
    wait(gd); C = dd(A)*dd(B); wait(gd);
    C = gather(C);
    alpha = 2.^( floor( log2( abs( max(dd(abs(hA)),[],2) ) ) ) );
    beta  = 2.^( floor( log2( abs( max(dd(abs(hB)),[],1) ) ) ) );
    err_max = zeros(length(N_s),1);
    err_min = zeros(length(N_s),1);
    est_max2 = zeros(length(N_s),1);
    est_min2 = zeros(length(N_s),1);
    est_max1 = zeros(length(N_s),1);
    est_min1 = zeros(length(N_s),1);
    for i = 1:length(N_s)
        fprintf('(phi, N) = (%.1f,%d)\n', phi, N_d(i));

        wait(gd); D = oz2(A,B,N_s(i),0); wait(gd);
        err = abs(double(gather(D)-C));
        [err_min(i), err_max(i)] = bounds(err,'all');

        p = mutually_coprime(256,N_s(i));
        [e,f,Cint,Cint2] = calc_Cmax(hA,hB,p);
        P = dd(1);
        for ip=1:length(p)
            P = P * dd(p(ip));
        end
        t = sqrt(2^(-5)./(P-1));
        rho = sum(dd(floor(p./2)));
        r32 = (1 + u32)*(N_s(i) + 2)*u64*rho*P + u32/2*P;
        % r32 = (1 + u32)*(N_s(i) + 2)*u64*rho*P + u32.*Cint2;
        est2 = t*(abs(hA)*v)*(beta .* sqrt(f)) ...
            + t*(alpha .* sqrt(e))*(v'*abs(hB)) ...
            + (k + r32).*(t^2*(alpha .* sqrt(e))*(beta .* sqrt(f)));
        [est_min2(i), est_max2(i)] = bounds(double(est2),'all');

        R32 = (1 + u32)*(N_s(i) + 2)*u64*rho*P + u32.*abs(Cint);
        est1 = t*(abs(hA)*v)*(beta .* sqrt(f)) ...
            + t*(alpha .* sqrt(e))*(v'*abs(hB)) ...
            + (k + R32).*(t^2*(alpha .* sqrt(e))*(beta .* sqrt(f)));
        [est_min1(i), est_max1(i)] = bounds(double(est1),'all');
    end

    C32 = A*B;
    err_32f = abs(double(gather(D)-C32));
    [err_32f_min, err_32f_max] = bounds(err_32f,'all');

    fig = figure; 
    plot(N_s,est_max1,'-r', ...
        N_s,est_min1,':r', ...
        N_s,est_max2,'-g', ...
        N_s,est_min2,':g', ...
        N_s,err_max,'-b', ...
        N_s,err_min,':b', ...
        N_s,err_32f_max*ones(size(N_s)),'--k', ...
        'LineWidth',1.5);
    set(gca,'YScale','Log','FontSize',14);
    xlim([min(N_s), max(N_s)]);
    xticks(N_s)
    grid on;
    xlabel('Number of moduli');
    ylabel('Absolute error')
    title("$\phi = " + phi + "$",'Interpreter','latex');
    legend("est\_max","est\_min", ...
        "est2\_max","est2\_min", ...
        "err\_max","err\_min", ...
        "err32\_max", ...
        'Location','best');
    p = char(string(phi*10 + 100));
    savefig(fig, "fig/s_phi=" + p(2:3));
    exportgraphics(fig,"fig/s_phi=" + p(2:3) + ".png",'Resolution',600);
end

end

%%
function [e,f,Cint,Cint2] = calc_Cmax(A,B,p)
maxA  = max(A,[],2,'ComparisonMethod','abs');
maxB  = max(B,[],1,'ComparisonMethod','abs');
ufpA  = ufp(maxA);
ufpB  = ufp(maxB);
sftA1 = 6 - (ufpA+1);
sftB1 = 6 - (ufpB+1);
Aint  = ceil(abs(A).*2.^sftA1);
Bint  = ceil(abs(B).*2.^sftB1);
Cint = Aint*Bint;
Cintu = double(flu(@() single(Cint)));
e = abs(max(Cintu,[],2,'ComparisonMethod','abs'));
f = abs(max(Cintu,[],1,'ComparisonMethod','abs'));

P = dd(1);
for i=1:length(p)
    P = P * dd(p(i));
end
log2e = double(log2(single(e)));
log2f = double(log2(single(f)));
log2P = log2(P-1)/2 - 0.5;
log2P = double(fld(@() single(log2P)));
half  = double(fld(@() single(-0.5/(1-2^-22))));
kA    = double(floor(fld(@() half*log2e+log2P )));
kB    = double(floor(fld(@() half*log2f+log2P )));
sftA2 = sftA1 + kA;
sftB2 = sftB1 + kB;
Aint  = fix(A.*2.^sftA2);
Bint  = fix(B.*2.^sftB2);
Cint2 = (2.^kA).*Cint.*(2.^kB);

d = dd.numSplit(5);
gd = gpuDevice();
wait(gd); C = dd(gpuArray(Aint))*dd(gpuArray(Bint)); wait(gd);
Cint = gather(C);
dd.numSplit(d);
end

%%
function x = fld(f)
feature('setround',-inf)
x  = f();
feature('setround',0.5)
end

function x = flu(f)
feature('setround',inf)
x  = f();
feature('setround',0.5)
end

%%
function B = ufp(A)
B = double(floor(log2(dd(abs(A)))));
B(B==-inf) = 0;
end

%%
function result = mutually_coprime(start, num)
if nargin < 2 num = 100;
end
result = zeros(1, num);
result(1)  = start;
len        = 1;
for i = (start - 1) : -1 : 3
    if mod(i,2) == 1
        if all(gcd(i, result(1 : len)) == 1)
            len = len + 1;
            result(len) = i;
            if len >= num
                break;
            end
        end
    end
end
end