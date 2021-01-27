import numpy as np
import casadi as cas


def CLA_and_CDA_fuse_hybrid(
        fuse_fineness_ratio,
        fuse_boattail_angle,
        fuse_TE_diameter,
        fuse_length,
        fuse_diameter,
        alpha,
        V,
        mach,
        rho,
        mu,
):
    alpha_rad = alpha * np.pi / 180
    sin_alpha = cas.sin(alpha_rad)
    cos_alpha = cas.cos(alpha_rad)

    """
    Lift of a truncated fuselage, following slender body theory.
    """
    separation_location = 0.3  # 0 for separation at trailing edge, 1 for separation at start of boat-tail. Calibrate this.
    diameter_at_separation = (1 - separation_location) * fuse_TE_diameter + separation_location * fuse_diameter

    fuse_TE_area = np.pi / 4 * diameter_at_separation ** 2
    CLA_slender_body = 2 * fuse_TE_area * alpha_rad  # Derived from FVA 6.6.5, Eq. 6.75

    """
    Crossflow force, following the method of Jorgensen, 1977: "Prediction of Static Aero. Chars. for Slender..."
    """
    V_crossflow = V * sin_alpha
    Re_crossflow = rho * cas.fabs(V_crossflow) * fuse_diameter / mu
    mach_crossflow = mach * cas.fabs(sin_alpha)
    eta = 1  # TODO make this a function of overall fineness ratio
    Cdn = 0.2  # Taken from suggestion for supercritical cylinders in Hoerner's Fluid Dynamic Drag, pg. 3-11
    S_planform = fuse_diameter * fuse_length
    CNA = eta * Cdn * S_planform * sin_alpha * cas.fabs(sin_alpha)
    CLA_crossflow = CNA * cos_alpha
    CDA_crossflow = CNA * sin_alpha

    CLA = CLA_slender_body + CLA_crossflow

    """
    Zero-lift_force drag_force
    Model derived from high-fidelity data at: C:\Projects\GitHub\firefly_aerodynamics\Design_Opt\studies\Circular Firefly Fuse CFD\Zero-Lift Drag
    """
    c = np.array([112.57153128951402720758778741583,
                  -7.1720570832587240417410612280946,
                  -0.01765596807595304351679033061373,
                  0.0026135564778264172743071913629365,
                  550.8775012129947299399645999074,
                  3.3166868391027000129156476759817,
                  11774.081980549422951298765838146,
                  3073258.204571904614567756652832,
                  0.0299,
                  ])  # coefficients
    fuse_Re = rho * cas.fabs(V) * fuse_length / mu
    CDA_zero_lift = (
            (
                    c[0] * cas.exp(c[1] * fuse_fineness_ratio) +
                    c[2] * fuse_boattail_angle +
                    c[3] * fuse_boattail_angle ** 2 +
                    c[4] * fuse_TE_diameter ** 2 +
                    c[5]
            ) / c[6] *
            (fuse_Re / c[7]) ** (-1 / 7) *
            (fuse_length * fuse_diameter) / c[8]
    )

    """
    Assumes a ring-shaped wake projection into the Trefftz plane with a sinusoidal circulation distribution.
    This results in a uniform grad(phi) within the ring in the Trefftz plane.
    Derivation at C:\Projects\GitHub\firefly_aerodynamics\Gists and Ideas\Ring Wing Potential Flow
    """
    fuse_oswalds_efficiency = 0.5
    CDiA_slender_body = CLA ** 2 / (
            diameter_at_separation ** 2 * np.pi * fuse_oswalds_efficiency) / 2  # or equivalently, use the next line
    # CDiA_slender_body = fuse_TE_area * alpha_rad ** 2 / 2

    CDA = CDA_crossflow + CDiA_slender_body + CDA_zero_lift

    return CLA, CDA


def CLA_and_CDA_nominal_fuse_CFD(alpha):
    """
    Calculated for fuse_fineness_ratio = 1.5, fuse_boattail_angle = 12, and fuse_TE_diameter = 20 mm.
    Lift: R^2 = 0.9961
    Drag: R^2 = 0.9876
    """
    CLA = (0.4605 * alpha) / 1e4
    CDA = (3.201 + 0.009754 * alpha ** 2) / 1e4
    return CLA, CDA


def Cf_flat_plate(Re_L):
    Re_L = cas.fabs(Re_L)
    # return 0.074 / Re_L ** 0.2  # Turbulent flat plate
    # return 0.02666 * Re_L ** -0.139  # Schlichting's model, roughly accounts for laminar part ("Boundary Layer Theory" 7th Ed., pg. 644)
    # return smoothmax(0.074 / Re_L ** 0.2 - 1742 / Re_L, 1.33 / Re_L ** 0.5, 1000)
    return cas.fmax(0.074 / Re_L ** 0.2 - 1742 / Re_L, 1.33 / Re_L ** 0.5)

def Cf_flat_plate_convex(Re_L):
    Re_L = cas.fabs(Re_L)
    # return 0.074 / Re_L ** 0.2  # Turbulent flat plate
    return 0.02666 * Re_L ** -0.139  # Schlichting's model, roughly accounts for laminar part ("Boundary Layer Theory" 7th Ed., pg. 644)
    # return smoothmax(0.074 / Re_L ** 0.2 - 1742 / Re_L, 1.33 / Re_L ** 0.5, 1000)
    # return cas.fmax(0.074 / Re_L ** 0.2 - 1742 / Re_L, 1.33 / Re_L ** 0.5)

def Cl_flat_plate(alpha, Re_c):
    Re_c = cas.fabs(Re_c)
    alpha_rad = alpha * np.pi / 180
    return 2 * np.pi * alpha_rad


def CL_over_Cl(AR, mach=0, sweep=0):
    """
    Returns the ratio of 3D lift_force coefficient (with compressibility) to 2D lift_force (incompressible) coefficient.
    :param AR:
    :param mach:
    :param sweep:
    :return:
    """
    beta = cas.sqrt(1 - mach ** 2)
    sweep_rad = sweep * np.pi / 180
    # return AR / (AR + 2) # Equivalent to equation in Drela's FVA in incompressible, 2*pi*alpha limit.
    # return AR / (2 + cas.sqrt(4 + AR ** 2))  # more theoretically sound at low AR
    eta = 0.95
    return AR / (
            2 + cas.sqrt(
        4 + (AR * beta / eta) ** 2 * (1 + (cas.tan(sweep_rad) / beta) ** 2)
    )
    )  # From Raymer, Sect. 12.4.1; citing DATCOM


def Cl_2412(alpha, Re_c):
    # A curve fit I did to a NACA 2412 airfoil, 2D XFoil data
    # Within -2 < alpha < 12 and 10^5 < Re_c < 10^7, has R^2 = 0.9892

    print("Warning: Cl_e216() recommended over Cl_2412(); those are MUCH more accurate fits.")

    return 0.2568 + 0.1206 * alpha - 0.002018 * alpha ** 2


def Cd_profile_2412(alpha, Re_c):
    # A curve fit I did to a NACA 2412 airfoil in incompressible flow.
    # Within -2 < alpha < 12 and 10^5 < Re_c < 10^7, has R^2 = 0.9713

    print("Warning: Cd_profile_e216() recommended over Cd_profile_2412(); those are MUCH more accurate fits.")

    Re_c = cas.fmax(Re_c, 1)
    log_Re = cas.log(Re_c)

    CD0 = -5.249
    Re0 = 15.61
    Re1 = 15.31
    alpha0 = 1.049
    alpha1 = -4.715
    cx = 0.009528
    cxy = -0.00588
    cy = 0.04838

    log_CD = CD0 + cx * (alpha - alpha0) ** 2 + cy * (log_Re - Re0) ** 2 + cxy * (alpha - alpha1) * (
            log_Re - Re1)  # basically, a rotated paraboloid in logspace
    CD = cas.exp(log_CD)

    return CD


def Cl_e216(alpha, Re_c):
    # A curve fit I did to a Eppler 216 (e216) airfoil, 2D XFoil data. Incompressible flow.
    # Within -2 < alpha < 12 and 10^4 < Re_c < 10^6, has R^2 = 0.9994
    # Likely valid from -6 < alpha < 12 and 10^4 < Re_c < Inf.
    # See: C:\Projects\GitHub\firefly_aerodynamics\Gists and Ideas\XFoil Drag Fitting\e216

    Re_c = cas.fmax(Re_c, 1)
    log10_Re = cas.log10(Re_c)

    # Coeffs
    a1l = 3.0904412662858878e-02
    a1t = 9.6452654383488254e-02
    a4t = -2.5633334023068302e-05
    asl = 6.4175433185427011e-01
    atr = 3.6775107602844948e-01
    c0l = -2.5909363461176749e-01
    c0t = 8.3824440586718862e-01
    ctr = 1.1431810545735890e+02
    ksl = 5.3416670116733611e-01
    rtr = 3.9713338634462829e+01
    rtr2 = -3.3634858542657771e+00
    xsl = -1.2220899840236835e-01

    a = alpha
    r = log10_Re

    Cl = (c0t + a1t * a + a4t * a ** 4) * 1 / (1 + cas.exp(ctr - rtr * r - atr * a - rtr2 * r ** 2)) + (
            c0l + a1l * a + asl / (1 + cas.exp(-ksl * (a - xsl)))) * (
                 1 - 1 / (1 + cas.exp(ctr - rtr * r - atr * a - rtr2 * r ** 2)))

    return Cl


def Cd_profile_e216(alpha, Re_c):
    # A curve fit I did to a Eppler 216 (e216) airfoil, 2D XFoil data. Incompressible flow.
    # Within -2 < alpha < 12 and 10^4 < Re_c < 10^6, has R^2 = 0.9995
    # Likely valid from -6 < alpha < 12 and 10^4 < Re_c < 10^6.
    # see: C:\Projects\GitHub\firefly_aerodynamics\Gists and Ideas\XFoil Drag Fitting\e216

    Re_c = cas.fmax(Re_c, 1)
    log10_Re = cas.log10(Re_c)

    # Coeffs
    a1l = 4.7167470806940448e-02
    a1t = 7.5663005080888857e-02
    a2l = 8.7552076545610764e-04
    a4t = 1.1220763679805319e-05
    atr = 4.2456038382581129e-01
    c0l = -1.4099657419753771e+00
    c0t = -2.3855286371940609e+00
    ctr = 9.1474872611212135e+01
    rtr = 3.0218483612170434e+01
    rtr2 = -2.4515094313899279e+00

    a = alpha
    r = log10_Re

    log10_Cd = (c0t + a1t * a + a4t * a ** 4) * 1 / (1 + cas.exp(ctr - rtr * r - atr * a - rtr2 * r ** 2)) + (
            c0l + a1l * a + a2l * a ** 2) * (1 - 1 / (1 + cas.exp(ctr - rtr * r - atr * a - rtr2 * r ** 2)))

    Cd = 10 ** log10_Cd

    return Cd


def Cd_wave_e216(Cl, mach, sweep=0.):
    """
    A curve fit I did to Eppler 216 airfoil data.
    Within -0.4 < CL < 0.75 and 0 < mach < ~0.9, has R^2 = 0.9982.
    See: C:\Projects\GitHub\firefly_aerodynamics\MSES Interface\analysis\e216
    :param Cl: Lift coefficient
    :param mach: Mach number
    :param sweep: Sweep angle, in deg
    :return: Wave drag coefficient.
    """

    mach = cas.fmax(mach, 0)
    sweep_rad = np.pi / 180 * sweep
    mach_perpendicular = mach * cas.cos(sweep_rad)  # Relation from FVA Eq. 8.176
    Cl_perpendicular = Cl / cas.cos(sweep_rad) ** 2  # Relation from FVA Eq. 8.177

    # Coeffs
    c0 = 7.2685945744797997e-01
    c1 = -1.5483144040727698e-01
    c3 = 2.1305118052118968e-01
    c4 = 7.8812272501525316e-01
    c5 = 3.3888938102072169e-03
    l0 = 1.5298928303149546e+00
    l1 = 5.2389999717540392e-01

    m = mach_perpendicular
    l = Cl_perpendicular

    Cd_wave = (cas.fmax(m - (c0 + c1 * cas.sqrt(c3 + (l - c4) ** 2) + c5 * l), 0) * (l0 + l1 * l)) ** 2

    return Cd_wave


def Cl_rae2822(alpha, Re_c):
    # A curve fit I did to a RAE2822 airfoil, 2D XFoil data. Incompressible flow.
    # Within -2 < alpha < 12 and 10^4 < Re_c < 10^6, has R^2 = 0.9857
    # Likely valid from -6 < alpha < 12 and 10^4 < Re_c < 10^6.
    # See: C:\Projects\GitHub\firefly_aerodynamics\Gists and Ideas\XFoil Drag Fitting\rae2822

    Re_c = cas.fmax(Re_c, 1)
    log10_Re = cas.log10(Re_c)

    # Coeffs
    a1l = 5.5686866813855172e-02
    a1t = 9.7472055628494134e-02
    a4l = -7.2145733312046152e-09
    a4t = -3.6886704372829236e-06
    atr = 8.3723547264375520e-01
    atr2 = -8.3128119739031697e-02
    c0l = -4.9103908291438701e-02
    c0t = 2.3903424824298553e-01
    ctr = 1.3082854754897108e+01
    rtr = 2.6963082864300731e+00

    a = alpha
    r = log10_Re

    Cl = (c0t + a1t * a + a4t * a ** 4) * 1 / (1 + cas.exp(ctr - rtr * r - atr * a - atr2 * a ** 2)) + (
            c0l + a1l * a + a4l * a ** 4) * (1 - 1 / (1 + cas.exp(ctr - rtr * r - atr * a - atr2 * a ** 2)))

    return Cl


def Cd_profile_rae2822(alpha, Re_c):
    # A curve fit I did to a RAE2822 airfoil, 2D XFoil data. Incompressible flow.
    # Within -2 < alpha < 12 and 10^4 < Re_c < 10^6, has R^2 = 0.9995
    # Likely valid from -6 < alpha < 12 and 10^4 < Re_c < Inf.
    # see: C:\Projects\GitHub\firefly_aerodynamics\Gists and Ideas\XFoil Drag Fitting\e216

    Re_c = cas.fmax(Re_c, 1)
    log10_Re = cas.log10(Re_c)

    # Coeffs
    at = 8.1034027621509015e+00
    c0l = -8.4296746456429639e-01
    c0t = -1.3700609138855402e+00
    kart = -4.1609994062600880e-01
    kat = 5.9510959342452441e-01
    krt = -7.1938030052506197e-01
    r1l = 1.1548628822014631e-01
    r1t = -4.9133662875044504e-01
    rt = 5.0070459892411696e+00

    a = alpha
    r = log10_Re

    log10_Cd = (c0t + r1t * (r - 4)) * (
                1 / (1 + cas.exp(kat * (a - at) + krt * (r - rt) + kart * (a - at) * (r - rt)))) + (
                           c0l + r1l * (r - 4)) * (
                           1 - 1 / (1 + cas.exp(kat * (a - at) + krt * (r - rt) + kart * (a - at) * (r - rt))))

    Cd = 10 ** log10_Cd

    return Cd


def Cd_wave_rae2822(Cl, mach, sweep=0.):
    """
    A curve fit I did to RAE2822 airfoil data.
    Within -0.4 < CL < 0.75 and 0 < mach < ~0.9, has R^2 = 0.9982.
    See: C:\Projects\GitHub\firefly_aerodynamics\MSES Interface\analysis\rae2822
    :param Cl: Lift coefficient
    :param mach: Mach number
    :param sweep: Sweep angle, in deg
    :return: Wave drag coefficient.
    """

    mach = cas.fmax(mach, 0)
    sweep_rad = np.pi / 180 * sweep
    mach_perpendicular = mach * cas.cos(sweep_rad)  # Relation from FVA Eq. 8.176
    Cl_perpendicular = Cl / cas.cos(sweep_rad) ** 2  # Relation from FVA Eq. 8.177

    # Coeffs
    c2 = 4.5776476424519119e+00
    mc0 = 9.5623337929607111e-01
    mc1 = 2.0552787101770234e-01
    mc2 = 1.1259268018737063e+00
    mc3 = 1.9538856688443659e-01

    m = mach_perpendicular
    l = Cl_perpendicular

    Cd_wave = cas.fmax(m - (mc0 - mc1 * cas.sqrt(mc2 + (l - mc3) ** 2)), 0) ** 2 * c2

    return Cd_wave


def Cd_wave_Korn(Cl, t_over_c, mach, sweep=0, kappa_A=0.95):
    """
    Wave drag_force coefficient prediction using the (very) low-fidelity Korn Equation method; derived in "Configuration Aerodynamics" by W.H. Mason, Sect. 7.5.2, pg. 7-18
    :param Cl: (2D) lift_force coefficient
    :param t_over_c: thickness-to-chord ratio
    :param sweep: sweep angle, in degrees
    :param kappa_A: Airfoil technology factor (0.95 for supercritical section, 0.87 for NACA 6-series)
    :return: Wave drag coefficient
    """
    mach = cas.fmax(mach, 0)
    sweep_rad = np.pi / 180 * sweep
    Mdd = kappa_A / cas.cos(sweep_rad) - t_over_c / cas.cos(sweep_rad) ** 2 - Cl / (10 * cas.cos(sweep_rad) ** 3)
    Mcrit = Mdd - (0.1 / 80) ** (1 / 3)
    # Cd_wave = 20 * cas.ramp(mach - Mcrit) ** 4
    Cd_wave = 20 * cas.if_else(mach > Mcrit, mach - Mcrit, 0) ** 4

    return Cd_wave


if __name__ == "__main__":
    # Run some checks
    import plotly.express as px

    # E216 checks
    alpha_inputs = np.linspace(-6, 12)
    Re_inputs = np.logspace(4, 6)
    alphas = []
    Res = []
    CLs = []
    CDs = []
    for alpha in alpha_inputs:
        for Re in Re_inputs:
            alphas.append(alpha)
            Res.append(Re)
            CLs.append(Cl_e216(alpha, Re))
            CDs.append(Cd_profile_e216(alpha, Re))
    px.scatter_3d(
        x=alphas,
        y=Res,
        z=CLs,
        size=np.ones_like(alphas),
        color=CLs,
        log_y=True,
        labels={"x": "alphas", "y": "Re", "z": "CL"}
    ).show()
    px.scatter_3d(
        x=alphas,
        y=Res,
        z=CDs,
        size=np.ones_like(alphas),
        color=CDs,
        log_y=True,
        labels={"x": "alphas", "y": "Re", "z": "CD"}
    ).show()
    px.scatter_3d(
        x=alphas,
        y=Res,
        z=np.array(CLs) / np.array(CDs),
        size=np.ones_like(alphas),
        color=np.array(CLs) / np.array(CDs),
        log_y=True,
        labels={"x": "alphas", "y": "Re", "z": "CL/CD"}
    ).show()
    
    # # rae2822 checks
    alpha_inputs = np.linspace(-6, 12)
    Re_inputs = np.logspace(4, 6)
    alphas = []
    Res = []
    CLs = []
    CDs = []
    for alpha in alpha_inputs:
        for Re in Re_inputs:
            alphas.append(alpha)
            Res.append(Re)
            CLs.append(Cl_rae2822(alpha, Re))
            CDs.append(Cd_profile_rae2822(alpha, Re))
    px.scatter_3d(
        x=alphas,
        y=Res,
        z=CLs,
        size=np.ones_like(alphas),
        color=CLs,
        log_y=True,
        labels={"x": "alphas", "y": "Re", "z": "CL"}
    ).show()
    px.scatter_3d(
        x=alphas,
        y=Res,
        z=CDs,
        size=np.ones_like(alphas),
        color=CDs,
        log_y=True,
        labels={"x": "alphas", "y": "Re", "z": "CD"}
    ).show()
    px.scatter_3d(
        x=alphas,
        y=Res,
        z=np.array(CLs) / np.array(CDs),
        size=np.ones_like(alphas),
        color=np.array(CLs) / np.array(CDs),
        log_y=True,
        labels={"x": "alphas", "y": "Re", "z": "CL/CD"}
    ).show()
    
    # # Cd_wave_e216 check
    # CL_inputs = np.linspace(-0.4, 1)
    # mach_inputs = np.linspace(0.3, 1)
    # CLs = []
    # machs = []
    # CD_waves = []
    # for CL in CL_inputs:
    #     for mach in mach_inputs:
    #         CLs.append(CL)
    #         machs.append(mach)
    #         CD_waves.append(Cd_wave_e216(CL, mach))
    # px.scatter_3d(
    #     x=CLs,
    #     y=machs,
    #     z=CD_waves,
    #     size=np.ones_like(CD_waves),
    #     color=CD_waves,
    #     title="E216 Fit",
    #     labels={"x": "CL", "y": "Mach", "z": "CD_wave"},
    #     range_z=(0, 200e-4)
    # ).show()

    # Cd_wave_rae2822 check
    # CL_inputs = np.linspace(-0.4, 1)
    # mach_inputs = np.linspace(0.3, 1)
    # CLs = []
    # machs = []
    # CD_waves = []
    # for CL in CL_inputs:
    #     for mach in mach_inputs:
    #         CLs.append(CL)
    #         machs.append(mach)
    #         CD_waves.append(Cd_wave_rae2822(CL, mach))
    # px.scatter_3d(
    #     x=CLs,
    #     y=machs,
    #     z=CD_waves,
    #     size=np.ones_like(CD_waves),
    #     color=CD_waves,
    #     title="RAE2822 Fit",
    #     labels={"x": "CL", "y": "Mach", "z": "CD_wave"},
    #     # range_z=(0, 200e-4)
    # ).show()



    # # Cd_wave_Korn check
    # CL_inputs = np.linspace(-0.4, 1)
    # mach_inputs = np.linspace(0.3, 1)
    # CLs = []
    # machs = []
    # CD_waves = []
    # for CL in CL_inputs:
    #     for mach in mach_inputs:
    #         CLs.append(CL)
    #         machs.append(mach)
    #         CD_waves.append(Cd_wave_Korn(CL, t_over_c=0.121, mach=mach, kappa_A=0.95))
    # px.scatter_3d(
    #     x=CLs,
    #     y=machs,
    #     z=CD_waves,
    #     size=np.ones_like(CD_waves),
    #     color=CD_waves,
    #     title="Korn Equation",
    #     labels={"x": "CL", "y": "Mach", "z": "CD_wave"},
    #     range_z=(0, 200e-4)
    # ).show()
