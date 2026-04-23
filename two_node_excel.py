## 原始two-node模型，每一组迭代60次，和two_node的区别是实现了表格批量计算和输出

import math
import pandas as pd
import os

# 获取当前脚本所在的目录，并设置为工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)   

print(f"脚本所在目录: {script_dir}")
print(f"当前工作目录: {os.getcwd()}")
print(f"目录中的文件: {os.listdir('.')}")

def calculate_comfort_parameters(input_row):
    """
    计算单行输入的舒适度参数
    """
    # 从输入行提取参数
    tdb = input_row['tdb']
    tr = input_row['tr']
    v = input_row['v']
    rh = input_row['rh']
    met = input_row['met']
    clo = input_row['clo']
    wme = input_row['wme']
    weight = input_row['weight']
    height = input_row['height']
    
    p_atmospheric = 101325
    body_position = "standing"

    max_skin_blood_flow = 90
    max_sweating = 500

    # 计算体表面积
    body_surface_area = 0.202 * (weight ** 0.425) * (height ** 0.725)

    # 下面为计算过程
    p_sat_torr = math.exp(18.6686 - 4030.183 / (tdb + 235.0))
    vapor_pressure = rh * p_sat_torr / 100

    # Initial variables as defined in the ASHRAE 55-2020
    air_speed = max(v, 0.1)
    k_clo = 0.25
    body_weight = 61.4  # body weight in kg
    met_factor = 58.2  # met conversion factor
    sbc = 0.000000056697  # Stefan-Boltzmann constant (W/m2K4)
    c_sw = 170  # driving coefficient for regulatory sweating
    c_dil = 200  # driving coefficient for vasodilation ashrae says 50 see page 195
    c_str = 0.5  # driving coefficient for vasoconstriction

    temp_skin_neutral = 33.7
    temp_core_neutral = 36.8
    alfa = 0.1
    temp_body_neutral = alfa * temp_skin_neutral + (1 - alfa) * temp_core_neutral
    skin_blood_flow_neutral = 6.3

    t_skin = temp_skin_neutral
    t_core = temp_core_neutral
    m_bl = skin_blood_flow_neutral

    # initialize some variables
    e_skin = 0.1 * met  # total evaporative heat loss, W
    q_sensible = 0  # total sensible heat loss, W
    w = 0  # skin wettedness
    _set = 0  # standard effective temperature
    e_rsw = 0  # heat lost by vaporization sweat
    e_diff = 0  # vapor diffusion through skin
    e_max = 0  # maximum evaporative capacity
    m_rsw = 0  # regulatory sweating
    e_res = 0  # latent heat loss due to respiration
    et = 0  # effective temperature
    e_req = 0  # evaporative heat loss required for tmp regulation
    r_ea = 0
    r_ecl = 0
    c_res = 0  # convective heat loss respiration

    pressure_in_atmospheres = p_atmospheric / 101325
    length_time_simulation = 60  # length time simulation
    n_simulation = 0

    r_clo = 0.155 * clo  # thermal resistance of clothing, C M^2 /W
    f_a_cl = 1.0 + 0.15 * clo  # increase in body surface area due to clothing
    lr = 2.2 / pressure_in_atmospheres  # Lewis ratio
    rm = (met - wme) * met_factor  # metabolic rate
    m = met * met_factor  # metabolic rate

    e_comfort = 0.42 * (rm - met_factor)  # evaporative heat loss during comfort
    if e_comfort < 0:
        e_comfort = 0

    if clo <= 0:
        w_max = 0.38 * pow(air_speed, -0.29)  # critical skin wettedness
        i_cl = 1.0  # permeation efficiency of water vapour through the clothing layer
    else:
        w_max = 0.59 * pow(air_speed, -0.08)  # critical skin wettedness
        i_cl = 0.45  # permeation efficiency of water vapour through the clothing layer

    # h_cc corrected convective heat transfer coefficient
    h_cc = 3.0 * pow(pressure_in_atmospheres, 0.53)
    # h_fc forced convective heat transfer coefficient, W/(m2 °C)
    h_fc = 8.600001 * pow((air_speed * pressure_in_atmospheres), 0.53)
    h_cc = max(h_cc, h_fc)
    if met > 0.85:
        h_c_met = 5.66 * (met - 0.85) ** 0.39
        h_cc = max(h_cc, h_c_met)

    h_r = 4.7  # linearized radiative heat transfer coefficient
    h_t = h_r + h_cc  # sum of convective and radiant heat transfer coefficient W/(m2*K)
    r_a = 1.0 / (f_a_cl * h_t)  # resistance of air layer to dry heat
    t_op = (h_r * tr + h_cc * tdb) / h_t  # operative temperature

    t_body = alfa * t_skin + (1 - alfa) * t_core  # mean body temperature, °C

    # respiration
    e_res = 0.0023 * m * (44.0 - vapor_pressure)  # latent heat loss due to respiration
    c_res = 0.0014 * m * (34.0 - tdb)  # sensible convective heat loss respiration

    # 按照时间迭代求解
    while n_simulation < length_time_simulation:
        # 开始循环

        iteration_limit = 150  # for following while loop
        # t_cl temperature of the outer surface of clothing
        t_cl = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)  # initial guess
        n_iterations = 0
        tc_converged = True
        
        while tc_converged:
            # 0.95 is the clothing emissivity from ASHRAE fundamentals Ch. 9.7 Eq. 35
            if body_position == "sitting":
                # 0.7 ratio between radiation area of the body and the body area
                h_r = 4.0 * 0.95 * sbc * ((t_cl + tr) / 2.0 + 273.15) ** 3.0 * 0.7
            else:  # if standing
                # 0.73 ratio between radiation area of the body and the body area
                h_r = 4.0 * 0.95 * sbc * ((t_cl + tr) / 2.0 + 273.15) ** 3.0 * 0.73
            h_t = h_r + h_cc
            r_a = 1.0 / (f_a_cl * h_t)
            t_op = (h_r * tr + h_cc * tdb) / h_t
            t_cl_new = (r_a * t_skin + r_clo * t_op) / (r_a + r_clo)
            if abs(t_cl_new - t_cl) <= 0.01:
                tc_converged = False
            t_cl = t_cl_new
            n_iterations += 1

            if n_iterations > iteration_limit:
                break
                
        q_sensible = (t_skin - t_op) / (r_a + r_clo)  # total sensible heat loss, W
        
        # hf_cs rate of energy transport between core and skin, W
        # 5.28 is the average body tissue conductance in W/(m2 C)
        # 1.163 is the thermal capacity of blood in Wh/(L C)
        hf_cs = (t_core - t_skin) * (5.28 + 1.163 * m_bl)
        s_core = m - hf_cs - e_res - c_res - wme  # rate of energy storage in the core
        s_skin = hf_cs - q_sensible - e_skin  # rate of energy storage in the skin
        tc_sk = 0.97 * alfa * body_weight  # thermal capacity skin
        tc_cr = 0.97 * (1 - alfa) * body_weight  # thermal capacity core
        d_t_sk = (s_skin * body_surface_area) / (tc_sk * 60.0)  # rate of change skin temperature °C per minute
        d_t_cr = (s_core * body_surface_area / (tc_cr * 60.0))  # rate of change core temperature °C per minute
        t_skin = t_skin + d_t_sk
        t_core = t_core + d_t_cr
        t_body = alfa * t_skin + (1 - alfa) * t_core
        # sk_sig thermoregulatory control signal from the skin
        sk_sig = t_skin - temp_skin_neutral
        warm_sk = (sk_sig > 0) * sk_sig  # vasodilation signal
        colds = ((-1.0 * sk_sig) > 0) * (-1.0 * sk_sig)  # vasoconstriction signal
        # c_reg_sig thermoregulatory control signal from the core, °C
        c_reg_sig = t_core - temp_core_neutral
        # c_warm vasodilation signal
        c_warm = (c_reg_sig > 0) * c_reg_sig
        # c_cold vasoconstriction signal
        c_cold = ((-1.0 * c_reg_sig) > 0) * (-1.0 * c_reg_sig)
        # bd_sig thermoregulatory control signal from the body
        bd_sig = t_body - temp_body_neutral
        warm_b = (bd_sig > 0) * bd_sig
        m_bl = (skin_blood_flow_neutral + c_dil * c_warm) / (1 + c_str * colds)
        if m_bl > max_skin_blood_flow:
            m_bl = max_skin_blood_flow
        if m_bl < 0.5:
             m_bl = 0.5
        m_rsw = c_sw * warm_b * math.exp(warm_sk / 10.7)  # regulatory sweating
        if m_rsw > max_sweating:
            m_rsw = max_sweating
        e_rsw = 0.68 * m_rsw  # heat lost by vaporization sweat
        r_ea = 1.0 / (lr * f_a_cl * h_cc)  # evaporative resistance air layer
        r_ecl = r_clo / (lr * i_cl)
        e_req = (rm - e_res - c_res - q_sensible)  # evaporative heat loss required for tmp regulation
        e_max = (math.exp(18.6686 - 4030.183 / (t_skin + 235.0)) - vapor_pressure) / (r_ea + r_ecl)
        p_rsw = e_rsw / e_max  # ratio heat loss sweating to max heat loss sweating
        w = 0.06 + 0.94 * p_rsw  # skin wetness
        e_diff = w * e_max - e_rsw  # vapor diffusion through skin
        if w > w_max:
            w = w_max
            p_rsw = w_max / 0.94
            e_rsw = p_rsw * e_max
            e_diff = 0.06 * (1.0 - p_rsw) * e_max
        if e_max < 0:
            e_diff = 0
            e_rsw = 0
            w = w_max
        e_skin = (e_rsw + e_diff)  # total evaporative heat loss sweating and vapor diffusion
        met_shivering = 19.4 * colds * c_cold  # met shivering W/m2
        m = rm + met_shivering
        alfa = 0.0417737 + 0.7451833 / (m_bl + 0.585417)
        
        Q_skin = q_sensible + e_skin  # total heat loss from skin, W

        n_simulation += 1   # 循环结束

    # p_s_sk saturation vapour pressure of water of the skin
    p_s_sk = math.exp(18.6686 - 4030.183 / (t_skin + 235.0))

    # standard environment - where _s at end of the variable names stands for standard
    h_r_s = h_r  # standard environment radiative heat transfer coefficient

    h_c_s = 3.0 * pow(pressure_in_atmospheres, 0.53)
    if met > 0.85:
        h_c_met = 5.66 * (met - 0.85) ** 0.39
        h_c_s = max(h_c_s, h_c_met)
    if h_c_s < 3.0:
        h_c_s = 3.0

    h_t_s = (h_c_s + h_r_s)  # sum of convective and radiant heat transfer coefficient W/(m2*K)
    r_clo_s = (1.52 / ((met - wme / met_factor) + 0.6944) - 0.1835)  # thermal resistance of clothing, °C M^2 /W
    r_cl_s = 0.155 * r_clo_s  # thermal insulation of the clothing in M2K/W
    f_a_cl_s = 1.0 + k_clo * r_clo_s  # increase in body surface area due to clothing
    f_cl_s = 1.0 / (1.0 + 0.155 * f_a_cl_s * h_t_s * r_clo_s)  # ratio of surface clothed body over nude body
    i_m_s = 0.45  # permeation efficiency of water vapour through the clothing layer
    i_cl_s = (i_m_s * h_c_s / h_t_s * (1 - f_cl_s) / (h_c_s / h_t_s - f_cl_s * i_m_s))  # clothing vapor permeation efficiency
    r_a_s = 1.0 / (f_a_cl_s * h_t_s)  # resistance of air layer to dry heat
    r_ea_s = 1.0 / (lr * f_a_cl_s * h_c_s)
    r_ecl_s = r_cl_s / (lr * i_cl_s)
    h_d_s = 1.0 / (r_a_s + r_cl_s)
    h_e_s = 1.0 / (r_ea_s + r_ecl_s)

    # calculate Standard Effective Temperature (SET)
    delta = 0.0001
    dx = 100.0
    set_old = round(t_skin - Q_skin / h_d_s, 2)
    while abs(dx) > 0.01:
        err_1 = (Q_skin- h_d_s * (t_skin - set_old)- w* h_e_s* (p_s_sk - 0.5 * (math.exp(18.6686 - 4030.183 / (set_old + 235.0)))))
        err_2 = (Q_skin- h_d_s * (t_skin - (set_old + delta))- w* h_e_s* (p_s_sk- 0.5 * (math.exp(18.6686 - 4030.183 / (set_old + delta + 235.0)))))
        _set = set_old - delta * err_1 / (err_2 - err_1)
        dx = _set - set_old
        set_old = _set

    # calculate Effective Temperature (ET)
    h_d = 1 / (r_a + r_clo)
    h_e = 1 / (r_ea + r_ecl)
    et_old = t_skin - Q_skin / h_d
    delta = 0.0001
    dx = 100.0
    while abs(dx) > 0.01:
        err_1 = (Q_skin- h_d * (t_skin - et_old)- w* h_e* (p_s_sk - 0.5 * (math.exp(18.6686 - 4030.183 / (et_old + 235.0)))))
        err_2 = (Q_skin- h_d * (t_skin - (et_old + delta))- w* h_e* (p_s_sk - 0.5 * (math.exp(18.6686 - 4030.183 / (et_old + delta + 235.0)))))
        et = et_old - delta * err_1 / (err_2 - err_1)
        dx = et - et_old
        et_old = et

    tbm_l = (0.194 / 58.15) * rm + 36.301  # lower limit for evaporative regulation
    tbm_h = (0.347 / 58.15) * rm + 36.669  # upper limit for evaporative regulation

    t_sens = 0.4685 * (t_body - tbm_l)  # predicted thermal sensation
    if (t_body >= tbm_l) & (t_body < tbm_h):
        t_sens = w_max * 4.7 * (t_body - tbm_l) / (tbm_h - tbm_l)
    elif t_body >= tbm_h:
        t_sens = w_max * 4.7 + 0.4685 * (t_body - tbm_h)

    disc = (4.7 * (e_rsw - e_comfort) / (e_max * w_max - e_comfort - e_diff))  # predicted thermal discomfort
    if disc <= 0:
        disc = t_sens

    # PMV Gagge
    pmv_gagge = (0.303 * math.exp(-0.036 * m) + 0.028) * (e_req - e_comfort - e_diff)

    # PMV SET
    dry_set = h_d_s * (t_skin - _set)
    e_req_set = rm - c_res - e_res - dry_set
    pmv_set = (0.303 * math.exp(-0.036 * m) + 0.028) * (e_req_set - e_comfort - e_diff)

    # Predicted Percent Satisfied With the Level of Air Movement"
    ps = 100 * (1.13 * (t_op ** 0.5) - 0.24 * t_op + 2.7 * (v ** 0.5) - 0.99 * v)

    # 汇总指标
    Q_sens = q_sensible + c_res
    Q_lat = e_skin + e_res
    Q_resp = c_res + e_res

    output = {
        "n_simulation": n_simulation,
        "Q_sens": Q_sens,
        "q_sensible": q_sensible,
        "c_res": c_res,
        "Q_lat": Q_lat,
        "e_skin": e_skin,
        "e_res": e_res,
        "e_rsw": e_rsw,
        "e_diff": e_diff,
        "Q_skin": Q_skin,
        "Q_resp": Q_resp,
        "e_max": e_max,
        "m_bl": m_bl,
        "m_rsw": m_rsw,
        "w": w,
        "w_max": w_max,
        "t_skin": t_skin,
        "t_core": t_core,
        "_set": _set,
        "et": et,
        "t_sens": t_sens,
        "disc": disc,
        "pmv_gagge": pmv_gagge,
        "pmv_set": pmv_set,
        "ps": ps,
        "r_clo_s": r_clo_s,
        "h_c_s": h_c_s
    }

    # 将结果圆整并转换为 Python float，避免 numpy 标量在 JSON 序列化时缺失
    def _round_to_float(val):
        try:
            return float(round(val, 3))
        except Exception:
            return val

    output = {key: _round_to_float(val) for key, val in output.items()}
    
    return output

# 主程序
def main():
    # 读取输入Excel文件
    try:
        input_df = pd.read_excel('input1.xlsx')
        print(f"成功读取输入文件，共{len(input_df)}行数据")
        print(f"列名: {list(input_df.columns)}")
    except FileNotFoundError:
        print("错误：找不到input1.xlsx文件")
        print("请确认:")
        print("1. 文件是否在当前目录")
        print("2. 文件名是否正确")
        print("3. 文件是否被其他程序占用")
        return
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return

    # 存储所有结果
    results = []

    # 对每一行数据进行计算
    for index, row in input_df.iterrows():
        try:
            print(f"正在计算第{index+1}行数据...")
            result = calculate_comfort_parameters(row)
            
            # 添加输入参数到结果中，便于追踪
            result.update({
                'tdb': row['tdb'],
                'tr': row['tr'], 
                'v': row['v'],
                'rh': row['rh'],
                'met': row['met'],
                'clo': row['clo'],
                'wme': row['wme'],
                'weight': row['weight'],
                'height': row['height']
            })
            
            results.append(result)
            print(f"第{index+1}行计算完成")
            
        except Exception as e:
            print(f"计算第{index+1}行时出错：{e}")
            # 添加错误信息
            error_result = {key: '计算错误' for key in [
                'n_simulation', 'Q_sens', 'q_sensible', 'c_res', 'Q_lat', 'e_skin',
                'e_res', 'e_rsw', 'e_diff', 'Q_skin', 'Q_resp', 'e_max', 'm_bl',
                'm_rsw', 'w', 'w_max', 't_skin', 't_core', '_set', 'et',
                't_sens', 'disc', 'pmv_gagge', 'pmv_set', 'ps', 'r_clo_s', 'h_c_s'
            ]}
            error_result.update({
                'tdb': row['tdb'],
                'tr': row['tr'],
                'v': row['v'],
                'rh': row['rh'],
                'met': row['met'],
                'clo': row['clo'],
                'wme': row['wme'],
                'weight': row['weight'],
                'height': row['height'],
                'error': str(e)
            })
            results.append(error_result)

    # 创建输出DataFrame
    output_df = pd.DataFrame(results)
    
    # 重新排列列的顺序，将输入参数放在前面
    input_columns = ['tdb', 'tr', 'v', 'rh', 'met', 'clo', 'wme', 'weight', 'height']
    output_columns = [col for col in output_df.columns if col not in input_columns and col != 'error']
    
    # 如果有错误列，也放在前面
    if 'error' in output_df.columns:
        final_columns = input_columns + ['error'] + output_columns
    else:
        final_columns = input_columns + output_columns
        
    output_df = output_df[final_columns]

    # 保存到输出Excel文件
    try:
        output_df.to_excel('output1.xlsx', index=False)
        print(f"结果已成功保存到output1.xlsx，共{len(output_df)}行数据")
    except Exception as e:
        print(f"保存输出文件时出错：{e}")

if __name__ == "__main__":
    main()
