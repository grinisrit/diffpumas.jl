# Assumed terrain scenarios, not measured DEM uncertainty. No production imports.
module TerrainStressGeometry
export terrain_lens_weight, stressed_elevation_map, path_snapshot

function terrain_lens_weight(x_m, y_m, height_m)
    height_m > 0 || return 0.0
    azimuth = mod(rad2deg(atan(x_m,y_m)),360.0)
    zenith = rad2deg(atan(hypot(x_m,y_m),height_m))
    # Smooth compact lens: zero value and slope on the declared angular edges.
    135.0 < azimuth < 180.0 && 40.0 < zenith < 55.0 || return 0.0
    sinpi((azimuth-135.0)/45.0)^2 * sinpi((zenith-40.0)/15.0)^2
end

function stressed_elevation_map(original, amplitude::Real, detector_elevation::Real)
    isfinite(amplitude) && abs(amplitude)<=0.05 || throw(ArgumentError("assumed stress amplitude must be within ±0.05"))
    result=deepcopy(original)
    amplitude==0 && return result # exact identity, including signed/rounding bits
    for iy in axes(original.data,1), ix in axes(original.data,2)
        x=1000*(original.x0+(ix-1)*original.dx)
        y=1000*(original.y0+(iy-1)*original.dy)
        height=original.data[iy,ix]-detector_elevation
        result.data[iy,ix] = original.data[iy,ix] + amplitude*height*terrain_lens_weight(x,y,height)
    end
    return result
end

path_snapshot(p)=(p.zenith,p.azimuth,[(s.cell_idx,s.distance) for s in p.segments],
    p.distance_in_volume,p.surface_distance,p.remaining_rock_distance,p.surface_exit_z,p.valid)
end
