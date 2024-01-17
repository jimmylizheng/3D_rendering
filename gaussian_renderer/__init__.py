#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, prePC: GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    screenspace_points_pre = torch.zeros_like(prePC.get_xyz, dtype=prePC.get_xyz.dtype, requires_grad=False, device="cuda") + 0

    all_xyz = torch.cat([pc._xyz, prePC._xyz], dim=0)
    all_opacity = torch.cat([pc._opacity, prePC._opacity], dim=0)
    
    screenspace_points_all = torch.cat([screenspace_points, screenspace_points_pre], dim=0)

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz
    # means2D = screenspace_points
    # opacity = pc.get_opacity
    means3D = all_xyz
    means2D = screenspace_points_all
    opacity = all_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        preCov = prePC.get_covariance(scaling_modifier)
        pcCov = pc.get_covariance(scaling_modifier)
        cov3D_precomp = torch.cat([pcCov, preCov], dim=0)
    else:
        scales = torch.cat([pc.get_scaling, prePC.get_scaling], dim=0)
        rotations = torch.cat([pc.get_rotation, prePC.get_rotation], dim=0)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            pcShs = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            preShs = prePC.get_features.transpose(1, 2).view(-1, 3, (prePC.max_sh_degree+1)**2)
            shs_view = torch.cat([pcShs, preShs], dim=0)

            
            pc_dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            pre_dir_pp = (prePC.get_xyz - viewpoint_camera.camera_center.repeat(prePC.get_features.shape[0], 1))
            dir_pp = torch.cat([pc_dir_pp, pre_dir_pp], dim=0)
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)


            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            pcShs = pc.get_features
            preShs = prePC.get_features
            shs = torch.cat([pcShs, preShs], dim=0)
            # shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
