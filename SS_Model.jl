"""
This file shows the code for S-S model.
The S-S model is based on Sobel and Schneider 2009 and 2013 papers, please refer to the papers for more details.
"""

## ----------------------------------------------
# model configuration
const integration_year_number = 10          # integrate how many years
const resolution_factor = 1                 # resolution factor. If less than 1, use reduced resolution to save time.
const dt=60*floor(1/resolution_factor+0.5)  # satisfies CFL requirement if wind speed is not insanely large
                                            # unit: s
Dir="D:/SS_model_output/YearCycle_ITCZDisplaced_MomentumInputCycle/LowRes_Kv100/" # path to save output files
const ITCZ_displaced = true                 # If true, annual mean ITCZ is located at 6 deg N.
                                            # ITCZ is controlled by radiative forcing
# ----------------------------------------------


## ----------------------------------------------
# Define constants
# For the meaning of constants, refer to Sobel and Schneider 2009
const β=2e-11
const kᵥ=7786*100
const ϵᵤ=1e-8
const Δz=60
const Δy=50
const T₀=300
const δ=4e3
const g=9.81
const H=16e3
const τ=37*24*3600
const y₁=9439e3
# ----------------------------------------------


# ----------------------------------------------
# define dimensions
if iseven(Int(floor(801*resolution_factor)))
    const y_Grids_for_V = Int(floor(801*resolution_factor)+1)
else
    const y_Grids_for_V = Int(floor(801*resolution_factor))
end
const y_Grids_for_UandT = y_Grids_for_V-1
const dy = 15751e3*2/y_Grids_for_UandT

const y_for_V=range(-15751e3, stop=15751e3,length=y_Grids_for_V)
const y_for_UandT=range(-15751e3+0.5*dy, step=dy ,length=y_Grids_for_UandT)

time = [0]
timestamp=0  # which day in the simulation
if ITCZ_displaced==ture 
    solar_insolation_degree=23.5*sin(2*pi*(timestamp%365-79)/365)+6
else
    solar_insolation_degree=23.5*sin(2*pi*(timestamp%365-79)/365)
end
# ----------------------------------------------


# ----------------------------------------------
# Define variables u, v and θ. (time, y)
u=fill(0.0,1,y_Grids_for_UandT)
v=fill(0.0,1,y_Grids_for_V)
# ----------------------------------------------


# ----------------------------------------------
# initialize theta_e and u.
# Define the radiative-convective equilibrium (RCE) temperature θₑ
θₑ=zeros(y_Grids_for_UandT)
θ00=330
u_initial=2*g*H*Δy/(1.6*T₀*β*y₁^2)
S=zeros(y_Grids_for_UandT)
u[1,:].=u_initial
for i=1:y_Grids_for_UandT
    if abs(y_for_UandT[i]-111e3*solar_insolation_degree) <= y₁
        θₑ[i]=θ00-Δy*((y_for_UandT[i]-111e3*solar_insolation_degree)/y₁).^2
        # u[:,i].=u_initial
    else
        θₑ[i]=θ00-Δy
    end
end
θ=reshape(θₑ,1,y_Grids_for_UandT)

function get_θₑ(timestamp)
    θₑ=zeros(y_Grids_for_UandT)
    θ00=330
    if ITCZ_displaced==ture 
        solar_insolation_degree=23.5*sin(2*pi*(timestamp%365-79)/365)+6
    else
        solar_insolation_degree=23.5*sin(2*pi*(timestamp%365-79)/365)
    end
    for i=1:y_Grids_for_UandT
        if abs(y_for_UandT[i]-111e3*solar_insolation_degree) <= y₁
            θₑ[i]=θ00-Δy*((y_for_UandT[i]-111e3*solar_insolation_degree)/y₁).^2
        else
            θₑ[i]=θ00-Δy
        end
    end
    return θₑ
end    
# ----------------------------------------------



# ----------------------------------------------
# Define necessary Functions
# Heaviside function
heaviside(x::Float64) = ifelse(x < 0, zero(x), ifelse(x > 0, one(x), oftype(x,0.5)))

# Get potential temperature
# According to the paper, the factor (pₛ/pₜ)^(R/cₚ) is 1.6
function get_θ(T)
    return 1.6*T
end

function get_T(θ)
    return θ/1.6
end

# The eddy momentum flux divergence function is based on the momentum budget in MERRA-2 reanalysis data.
# Please refer to Zhang and Lutsko 2022 for details.
function get_S(timestamp,i)
    day_of_year=timestamp%365
    if day_of_year<60 || day_of_year>285 
        S=-3e-6*exp(-(y_for_UandT[i]/(3*111e3))^2)+3e-6*exp(-((y_for_UandT[i]-15*111e3)/(3*111e3))^2)+0.2e-6*exp(-((y_for_UandT[i]+15*111e3)/(3*111e3))^2)
    elseif day_of_year>155 && day_of_year<260
        S=-2.8e-6*exp(-(y_for_UandT[i]/(3*111e3))^2)+1.6e-6*exp(-((y_for_UandT[i]+15*111e3)/(3*111e3))^2)+1.6e-6*exp(-((y_for_UandT[i]-15*111e3)/(3*111e3))^2)
    else 
        S=-1.6e-6*exp(-(y_for_UandT[i]/(3*111e3))^2)+0.6e-6*exp(-((y_for_UandT[i]+15*111e3)/(3*111e3))^2)+0.6e-6*exp(-((y_for_UandT[i]-15*111e3)/(3*111e3))^2)
    end
    return S*0.8
end

# First PDE about u:
function get_du(u,v,θ,timestamp,dt)
    adv=fill(0.0,y_Grids_for_V)
    adv_for_UandT=fill(0.0,y_Grids_for_UandT)
    u_grad=fill(0.0,y_Grids_for_V)
    v_grad=fill(0.0,y_Grids_for_UandT)
    Hvy=fill(0.0,y_Grids_for_UandT) # Heaviside of partial v / partial y
    beta=fill(0.0,y_Grids_for_V)
    beta_for_UandT=fill(0.0,y_Grids_for_UandT)
    # damping=fill(0.0,y_Grids_for_UandT)    
    # Damping for u is not required.
    F=ϵᵤ*u
    for i=1:y_Grids_for_UandT
        v_grad[i]=(v[i+1]-v[i])/dy
        if i!=y_Grids_for_UandT
            u_grad[i+1]=(u[i+1]-u[i])/dy
        end
        adv=v.*u_grad
        beta[i+1]=β*y_for_V[i+1]*v[i+1] 
        Hvy[i]=heaviside(v_grad[i])
    end
    for i=1:y_Grids_for_UandT
        adv_for_UandT[i]=(adv[i]+adv[i+1])/2
        beta_for_UandT[i]=(beta[i]+beta[i+1])/2
        if abs(y_for_UandT[i]-111e3*solar_insolation_degree) <= y₁
            S[i]=get_S(timestamp,i)
        end
        
    end
    return (beta_for_UandT-adv_for_UandT-Hvy.*v_grad.*u-F-S)*dt
end

# Second PDE about v 
function get_dv(u,v,θ,dt)
    T=get_T(θ)
    beta=fill(0.0,y_Grids_for_V)
    T_grad=fill(0.0,y_Grids_for_V)
    damping=fill(0.0,y_Grids_for_V)
    for i=1:y_Grids_for_UandT-2
        T_grad[i+1]=(T[i+1]-T[i])/dy
        beta[i+1]=β*y_for_V[i+1]*(u[i]+u[i+1])*0.5
        damping[i+1]=(v[i+2]+v[i]-2*v[i+1])/(dy^2)
    end
    return (-beta-g*H*T_grad/T₀+kᵥ*damping)*0.5*dt
end


# Third PDE about θ
function get_dθ(u,v,θ,timestamp,dt)
    v_grad=fill(0.0,y_Grids_for_UandT)
    if ITCZ_displaced==ture 
        solar_insolation_degree=23.5*sin(2*pi*(timestamp%365-79)/365)+6
    else
        solar_insolation_degree=23.5*sin(2*pi*(timestamp%365-79)/365)
    end
    for i=1:y_Grids_for_UandT
        if abs(y_for_UandT[i]-111e3*solar_insolation_degree) <= y₁
            v_grad[i]=(v[i+1]-v[i])/dy
        else 
            v_grad[i]=0
        end
    end
    θₑ=get_θₑ(timestamp)
    return ((θₑ.-θ)/τ-δ*Δz*v_grad/H)*dt  # need to prescribe θₑ
end
# ----------------------------------------------


# ----------------------------------------------
# Define function for integration
# Function of Eular scheme. Only used for the first calculation.
function Euler(u,v,θ,get_du,get_dv,get_dθ,timestamp,dt)
    return u+get_du(u,v,θ,timestamp,dt), v+get_dv(u,v,θ,dt), θ+get_dθ(u,v,θ,timestamp,dt)
end

# leapfrog function. 
# a_n+1 = a_n-1 + F(a_n) * 2 * dt
function leapfrog(u,u_before,v,v_before,θ,θ_before,get_du,get_dv,get_dθ,timestamp,dt)
    return u_before+get_du(u,v,θ,timestamp,2*dt), v_before+get_dv(u,v,θ,2*dt), θ_before+get_dθ(u,v,θ,timestamp,2*dt)
end
# ----------------------------------------------


# ----------------------------------------------
# core code for the integration
timesteps=Int(86400*365*integration_year_number/dt)

u_thisstep=u[end,:]
v_thisstep=v[end,:]
θ_thisstep=θ[end,:]
u_before, v_before, θ_before=Euler(u_thisstep,v_thisstep,θ_thisstep,get_du,get_dv,get_dθ,timestamp,-dt)
for i = 1:timesteps
    u_thisstep[1]=0
    u_thisstep[end]=0
    v_thisstep[1]=0
    v_thisstep[end]=0
    global timestamp
    u_after,v_after,θ_after=leapfrog(u_thisstep,u_before,v_thisstep,v_before,θ_thisstep,θ_before,get_du,get_dv,get_dθ,timestamp,dt)
    global u_before=u_thisstep+0.04*(u_after+u_before-2*u_thisstep)
    global v_before=v_thisstep+0.04*(v_after+v_before-2*v_thisstep)
    global θ_before=θ_thisstep+0.04*(θ_after+θ_before-2*θ_thisstep)
    global u_thisstep=u_after
    global v_thisstep=v_after
    global θ_thisstep=θ_after
    time_end=time[end]
    if isempty(u)
        u=reshape(u_thisstep,1,y_Grids_for_UandT)
        v=reshape(v_thisstep,1,y_Grids_for_V)
        θ=reshape(θ_thisstep,1,y_Grids_for_UandT)
    else
        append!(time,[time_end+dt])
        global u=vcat(u,reshape(u_thisstep,1,y_Grids_for_UandT))
        global v=vcat(v,reshape(v_thisstep,1,y_Grids_for_V))
        global θ=vcat(θ,reshape(θ_thisstep,1,y_Grids_for_UandT))
    end

    if mod((i+1)*dt,86400)==0
        stamp=lpad(Int((i+1)*dt/(86400)),5,"0")
        println("Day $stamp finished.")
        timestamp+=1
        if i>= (integration_year_number-1)*timesteps/integration_year_number
            using NCDatasets
            ds_u = NCDataset("$Dir/u_output_Day$stamp.nc","c")
            ds_v = NCDataset("$Dir/v_output_Day$stamp.nc","c")
            ds_θ = NCDataset("$Dir/pot.temp_Day$stamp.nc","c")
            # Define the dimension
            defDim(ds_u,"time",Inf)
            defDim(ds_u,"y",y_Grids_for_UandT)
            defDim(ds_v,"time",Inf)
            defDim(ds_v,"y",y_Grids_for_V)
            defDim(ds_θ,"time",Inf)
            defDim(ds_θ,"y",y_Grids_for_UandT)
            
            # Define a global attribute, the long name of the file.
            ds_u.attrib["title"] = "Zonal wind speed"
            ds_v.attrib["title"] = "Meridional wind speed"
            ds_θ.attrib["title"] = "Potential temperature"

            # Define the variables velocity
            variable_u = defVar(ds_u,"U",Float64,("y","time"))
            variable_uy= defVar(ds_u,"y",Float64,("y",))
            variable_ut= defVar(ds_u,"time",Float64,("time",))
            variable_v = defVar(ds_v,"V",Float64,("y","time"))
            variable_vy= defVar(ds_v,"Y",Float64,("y",))
            variable_vt= defVar(ds_v,"time",Float64,("time",))
            variable_θ = defVar(ds_θ,"θ",Float64,("y","time"))
            variable_θy= defVar(ds_θ,"Y",Float64,("y",))
            variable_θt= defVar(ds_θ,"time",Float64,("time",))

            # write a the complete data set
            variable_u[:,:] = permutedims(u, [2,1])
            variable_ut[:]=time
            variable_uy[:]=y_for_UandT
            variable_v[:,:] = permutedims(v, [2,1])
            variable_vt[:]=time
            variable_vy[:]=y_for_V
            variable_θ[:,:] = permutedims(θ, [2,1])
            variable_θt[:]=time
            variable_θy[:]=y_for_UandT

            # write attributes
            variable_u.attrib["units"] = "m/s"
            variable_v.attrib["units"] = "m/s"
            variable_θ.attrib["units"] = "K"
            variable_uy.attrib["units"]= "m"
            variable_vy.attrib["units"]= "m"
            variable_θy.attrib["units"]= "m"
            variable_ut.attrib["units"]= "seconds since 0000-01-01 00:00:00.0"
            variable_vt.attrib["units"]= "seconds since 0000-01-01 00:00:00.0"
            variable_θt.attrib["units"]= "seconds since 0000-01-01 00:00:00.0"
            variable_ut.attrib["calendar"]= "noleap"
            variable_vt.attrib["calendar"]= "noleap"
            variable_θt.attrib["calendar"]= "noleap"

            close(ds_u)
            close(ds_v)
            close(ds_θ)
        end

        global time=[time_end+2*dt]
        global u=[]
        global v=[]
        global θ=[]
    end
end
# ----------------------------------------------

timeend=time[end]/(86400*365)
println("We have already run $timeend years.")
