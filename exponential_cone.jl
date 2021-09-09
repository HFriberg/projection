using LinearAlgebra
using Formatting
using Quadmath

function hfun(v0::Array{real,1}, rho::real) where real<:Real
    t0,s0,r0 = v0;
    exprho = exp(rho);
    expnegrho = exp(-rho);
    f  = ((rho-1)*r0+s0)*exprho -     (r0-rho*s0)*expnegrho - (rho*(rho-1)+1)*t0;
    df =     (rho*r0+s0)*exprho + (r0-(rho-1)*s0)*expnegrho -       (2*rho-1)*t0;
    return [f,df];
end

function rootsearch_bn(fun::Function, farg::Any, xl::real, xh::real, x0::real) where real<:Real
    EPS = eps(real);

    @assert xl<=x0<=xh

    local xx;

    while true
        f = fun(farg,x0)[1];
        if( f < 0.0 )
            xl = x0;
        else
            xh = x0;
        end

        xx = 0.5*(xl+xh)
        #println(["BN",xx,xl,xh,f])

        if( abs(xx-x0) <= EPS*max(1.,abs(xx)) || xx==xl || xx==xh )
            break;
        end

        x0 = xx;
    end

    return xx;
end

#
# Newton root search for strictly increasing functions
#
function rootsearch_ntinc(fun::Function, farg::Any, xl::real, xh::real, x0::real) where real<:Real
    EPS = eps(real);
    DFTOL = EPS^(6/7);
    MAXITER = 20;
    LODAMP = parse(real,"0.05")
    HIDAMP = parse(real,"0.95")

    @assert xl<=x0<=xh

    xx = x0;
    converged = false;

    for i = 1:MAXITER
        f,df = fun(farg,x0)
        if( f < 0.0 )
            xl = x0;
        else
            xh = x0;
        end

        if( xh<=xl )
            converged = true;
            break;
        end

        if( isfinite(f) && df >= DFTOL )
            xx = x0 - f/df;
        else
            break;
        end

        if( abs(xx-x0) <= EPS*max(1.,abs(xx)) )
            converged = true;
            break;
        end

        # Dampened steps to boundary
        if( xx>=xh )
            x0 = min(LODAMP*x0+HIDAMP*xh, xh);
        elseif ( xx<=xl )
            x0 = max(LODAMP*x0+HIDAMP*xl, xl);
        else
            x0 = xx;
        end

        #println(["NT",x0,xl,xh,f])
    end

    if( converged )
        return max(xl,min(xh,xx));
    else
        return rootsearch_bn(fun,farg,xl,xh,x0);
    end
end

function projheu_primalexpcone(v0::Array{real,1}) where real<:Real
    t0,s0,r0 = v0

    # perspective boundary
    vp = [max(t0,0), 0.0, min(r0,0)] 
    dist = norm(vp-v0,2)

    # perspective interior
    if s0 > 0.0
        tp = max(t0, s0*exp(r0/s0))
        newdist = tp-t0
        if newdist < dist
            vp = [tp, s0, r0]
            dist = newdist
        end
    end

    return [vp,dist]
end

function projheu_polarexpcone(v0::Array{real,1}) where real<:Real
    t0,s0,r0 = v0

    # perspective boundary
    vd = [min(t0,0), min(s0,0), 0.0]
    dist = norm(vd-v0,2)

    # perspective interior
    if r0 > 0.0
        td = min(t0, -r0*exp(s0/r0-1))
        newdist = t0-td
        if newdist < dist
            vd = [td, s0, r0]
            dist  = newdist
        end
    end
    
    return [vd,dist]
end

function projsol_primalexpcone(v0::Array{real,1}, rho::real) where real<:Real
    t0,s0,r0 = v0
    local vp,dist

    linrho = ((rho-1)*r0+s0)
    exprho = exp(rho)
    if (linrho>0) && isfinite(exprho)
        quadrho=rho*(rho-1)+1
        vp=[exprho,1,rho]*linrho/quadrho
        dist=norm(vp-v0,2)
    else
        vp = [Inf,0.0,0.0]
        dist = Inf
    end
    
    return [vp,dist]
end

function projsol_polarexpcone(v0::Array{real,1}, rho::real) where real<:Real
    t0,s0,r0 = v0
    local vd,dist

    linrho = (r0-rho*s0)
    exprho = exp(-rho)
    if (linrho>0) && isfinite(exprho)
        quadrho=rho*(rho-1)+1
        vd=[-exprho,1-rho,1]*linrho/quadrho
        dist=norm(vd-v0,2)
    else
        vd = [-Inf,0.0,0.0]
        dist = Inf
    end
    
    return [vd,dist]
end

function ppsi(v0::Array{real,1}) where real<:Real
    t0,s0,r0 = v0
    
    # two expressions for the same to avoid catastrophic cancellation
    if (r0 > s0)
        psi = (r0-s0 + sqrt(r0^2 + s0^2 - r0*s0)) / r0
    else
        psi = -s0 / (r0-s0 - sqrt(r0^2 + s0^2 - r0*s0))
    end
    
    return ((psi-1)*r0 + s0)/(psi*(psi-1) + 1)
end

function pomega(rho::real) where real<:Real
    val = exp(rho)/(rho*(rho-1)+1)
    if rho < 2.0
        val = min(val, exp(parse(real,"2"))/3)
    end
   
    return val
end

function dpsi(v0::Array{real,1}) where real<:Real
    t0,s0,r0 = v0
    
    # two expressions for the same to avoid catastrophic cancellation
    if( s0 > r0 )
        psi = (r0 - sqrt(r0^2 + s0^2 - r0*s0)) / s0
    else
        psi = (r0 - s0) / (r0 + sqrt(r0^2 + s0^2 - r0*s0))
    end
    
    res = (r0 - psi*s0)/(psi*(psi-1) + 1)
    return res
end

function domega(rho::real) where real<:Real
    val = -exp(-rho)/(rho*(rho-1)+1)
    if rho > -1.0
        val = max(val, -exp(one(real))/3)
    end
   
    return val
end

function searchbracket(v0::Array{real,1}, pdist::real, ddist::real) where real<:Real
    t0,s0,r0 = v0
    baselow,baseupr = real(-Inf),real(Inf)
    low,upr = real(-Inf),real(Inf)

    Dp = sqrt(pdist^2 - min(s0,0)^2)
    Dd = sqrt(ddist^2 - min(r0,0)^2)

    if (t0>0)
        tpl    = t0
        curbnd = log(tpl/ppsi(v0))
        low    = max(low,curbnd);
    end
    
    if (t0<0)
        tdu    = t0
        curbnd = -log(-tdu/dpsi(v0));
        upr    = min(upr, curbnd);
    end

    if (r0>0)
        baselow = 1-s0/r0
        low     = max(low, baselow)

        tpu    = max(1e-12, min(Dd, Dp+t0))
        palpha = low
        curbnd = max(palpha, baselow + tpu/r0/pomega(palpha));
        upr    = min(upr, curbnd);
    end

    if (s0>0)
        baseupr = r0/s0
        upr     = min(upr, baseupr)

        tdl    = -max(1e-12, min(Dp, Dd-t0))
        dalpha = upr
        curbnd = min(dalpha, baseupr - tdl/s0/domega(dalpha))
        low    = max(low, curbnd)
    end

    @assert baselow <= baseupr
    @assert isfinite(low)
    @assert isfinite(upr)

    # Guarantee valid bracket
    low,upr = min(low,upr),max(low,upr)
    low,upr = clamp(low,baselow,baseupr),clamp(upr,baselow,baseupr)
    if low!=upr
        fl = hfun(v0,low)[1]
        fu = hfun(v0,upr)[1]

        if !(fl*fu < 0)
            if (abs(fl)<abs(fu) || isnan(fl))
                upr = low;
            else
                low = upr;
            end
        end
    end
    
    return [low,upr]
end

function proj_primalexpcone(v0::Array{real,1}) where real<:Real
    TOL = eps(real)^(2/3)
    t0,s0,r0 = v0
    
    vp,pdist = projheu_primalexpcone(v0)
    vd,ddist = projheu_polarexpcone(v0)
    
    # Skip root search if presolve rules apply
    # or optimality conditions are satisfied
    #
    if !( (s0<=0 && r0<=0) || min(pdist,ddist)<=TOL || ( norm(vp+vd-v0,Inf)<=TOL && dot(vp,vd)<=TOL ) )

        xl,xh = searchbracket(v0,pdist,ddist)
        rho   = rootsearch_ntinc(hfun,v0,xl,xh,0.5*(xl+xh))

        vp1,pdist1 = projsol_primalexpcone(v0,rho)
        if (pdist1 <= pdist)
            vp,pdist = vp1,pdist1
        end
        
        vd1,ddist1 = projsol_polarexpcone(v0,rho)
        if (ddist1 <= ddist)
            vd,ddist = vd1,ddist1
        end
    end
    
    return [vp,vd]
end

function abserr(v0::Array{real,1}, vp::Array{real,1}, vd::Array{real,1}) where real<:Real
    return [norm(vp + vd - v0,2), 
            abs(dot(vp,vd))];
end

function relerr(v0::Array{real,1}, vp::Array{real,1}, vd::Array{real,1}) where real<:Real
    return abserr(v0,vp,vd) / max(1.0,norm(v0,2));
end

function solutionreport(v0::Array{real,1}, vp::Array{real,1}, vd::Array{real,1}) where real<:Real
    println("abserr=$(fmt.("1.1e",Float64.(abserr(v0,vp,vd))))\nrelerr=$(fmt.("1.1e",Float64.(relerr(v0,vp,vd))))\n  v0=$(Float64.(v0))\n  vp=$(Float64.(vp)) in primal\n  vd=$(Float64.(vd)) in polar\n")
end

function test(real::Type)
    low=-20;
    upr= 21;
    domain = [-exp.(low:upr); 0.0; exp.(low:upr)]

    maxerr1 = 0.0;
    maxerr2 = 0.0;

    # precompile and extract stats
    begin
        for t0 in domain
            for s0 in domain
                for r0 in domain
                    v0 = Array{real,1}( [t0, s0, r0] );
                    try
                        vp,vd = proj_primalexpcone(v0)
                        curerr1,curerr2 = relerr(v0,vp,vd);
                        maxerr1 = max(maxerr1, curerr1)
                        maxerr2 = max(maxerr2, curerr2)
                    catch
                        println("ERROR: $v0")
                        rethrow()
                    end
                end
            end
        end
    end

    # benchmark time
    time = @elapsed begin
        for t0 in domain
            for s0 in domain
                for r0 in domain
                    v0 = Array{real,1}( [t0, s0, r0] );
                    vp,vd = proj_primalexpcone(v0)
                end
            end
        end
    end

    numproj=length(domain)^3
    return (maxerr1=maxerr1, maxerr2=maxerr2, numproj=numproj, totaltime=time, avgtime=time/numproj);
end

function testCOSMO()
    low=-20;
    upr= 21;
    domain = [-exp.(low:upr); 0.0; exp.(low:upr)]

    maxerr1 = 0.0;
    maxerr2 = 0.0;

    COSMOpexp = COSMO.ExponentialCone();

    function COSMOproj_primalexpcone(v0)
        # compute vp using COSMO
        vp = reverse(v0);
        COSMO.project!(vp, COSMOpexp);
        vp=reverse(vp)

        # fill in missing details of vd (computed in high precision for an unbiased comparison)
        _,vd = proj_primalexpcone(Array{Float128,1}(v0))
        vd = Float64.(vd)

        return [vp,vd]
    end

    #v0 = Array{Float64,1}( [1, 1, 1] );
    #vp,vd = COSMOproj_primalexpcone(v0)
    #solutionreport(v0,vp,vd)

    # precompile and extract stats
    begin
        for t0 in domain
            for s0 in domain
                for r0 in domain
                    v0 = Array{Float64,1}( [t0, s0, r0] );
                    try
                        vp,vd = COSMOproj_primalexpcone(v0)
                        curerr1,curerr2 = relerr(v0,vp,vd);
                        maxerr1 = max(maxerr1, curerr1)
                        maxerr2 = max(maxerr2, curerr2)
                    catch
                        println("ERROR: $v0")
                        rethrow()
                    end
                end
            end
        end
    end

    # benchmark time
    time = @elapsed begin
        for t0 in domain
            for s0 in domain
                for r0 in domain
                    v0 = Array{Float64,1}( [t0, s0, r0] );
                    COSMO.project!(v0, COSMOpexp);
                end
            end
        end
    end

    numproj=length(domain)^3
    return (maxerr1=maxerr1, maxerr2=maxerr2, numproj=numproj, totaltime=time, avgtime=time/numproj);
end

v0 = Array{Float64,1}( [1, 1, 1] );
vp,vd = proj_primalexpcone(v0)
solutionreport(v0,vp,vd)

if true
    println(test(Float64));
end

if true
    using COSMO;
    println(testCOSMO());
end
