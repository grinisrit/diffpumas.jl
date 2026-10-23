using DiffPumas
using DiffPumas.Physics: property_straggling, property_cross_section
p=load_physics(ARGS[1]);p===nothing && error("Unreadable dump")
open(ARGS[2],"w") do io
    println(io,"material,energy,property,value")
    for name in ["StandardRock","Air"], energy in [10.,100.,1000.,10000.,100000.]
        material=get_material_index(p,name)
        for (field,value) in [
            ("stopping_power_0",property_stopping_power(p,ENERGY_LOSS_CSDA,material,energy)),
            ("stopping_power_1",property_stopping_power(p,ENERGY_LOSS_MIXED,material,energy)),
            ("range_0",property_range(p,ENERGY_LOSS_CSDA,material,energy)),
            ("range_1",property_range(p,ENERGY_LOSS_MIXED,material,energy)),
            ("transport_path_1",property_transport_path(p,material,energy)),
            ("proper_time_0",property_proper_time(p,ENERGY_LOSS_CSDA,material,energy)),
            ("energy_straggling",property_straggling(p,material,energy)),
            ("cross_section",property_cross_section(p,material,energy))]
            println(io,join((name,energy,field,value),','))
        end
    end
end
