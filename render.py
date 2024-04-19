from typing import Optional, List
import pathlib
import argparse
import bpy

def render(scene_file: pathlib.Path,
           object_file: pathlib.Path,
           collider_file: pathlib.Path,
           output_file: pathlib.Path,
           render_still: Optional[int] = None,
           render_animation: Optional[List[int]] = None,
           num_samples: Optional[int] = None):
    """Render a contact interaction with Blender, provided the vertex position sequences of 
       the deformable object and rigid collider, together with the corresponding blender scene.

    Keyword arguments:

    scene_file:       Path to blender scene file of the interaction (.blend).

    object_file:      Path to vertex position sequence file of the deformable object (.pc2).

    collider_file:    Path to vertex position sequence file of the rigid collider (.pc2).

    output_file:      Path to desired output file (.png for stills or .mkv for animations).

    render_still:     Optional number [frame] to render a still. No still rendered by default.

    render_animation: Optional list of numbers [frame_start, frame_end, frames_per_second] to render an animation. 
                      No animation rendered by default.
                      
    num_samples:      Optional number of samples used for rendering. Use small values (1-4) for fast and low quality preview.      
    """

    bpy.ops.wm.open_mainfile('EXEC_DEFAULT', filepath = str(scene_file.absolute()), load_ui=False)

    bpy.data.objects['Object'].modifiers['MeshCache'].filepath = str(object_file.absolute())
    bpy.data.objects['Collider'].modifiers['MeshCache'].filepath = str(collider_file.absolute())

    scene = bpy.data.scenes['Scene']

    scene.render.filepath = str(output_file.absolute())

    if num_samples is not None:
        scene.eevee.taa_render_samples = num_samples
        
    if render_still is not None:
        scene.frame_current = render_still
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.film_transparent = True

        bpy.ops.render.render('EXEC_DEFAULT', write_still=True)

    if render_animation is not None:
        scene.frame_start = render_animation[0]
        scene.frame_end = render_animation[1]
        scene.render.fps = render_animation[2]
        scene.render.fps_base = 1.0
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.constant_rate_factor = 'HIGH'
        scene.render.film_transparent = False

        bpy.ops.render.render('EXEC_DEFAULT', animation=True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--scene_file', type=pathlib.Path, required=True,
                        help='Path to blender scene file of the interaction (.blend)')
    parser.add_argument('--object_file', type=pathlib.Path, required=True,
                        help='Path to vertex position sequence file of the deformable object (.pc2)')
    parser.add_argument('--collider_file', type=pathlib.Path, required=True,
                        help='Path to vertex position sequence file of the rigid collider (.pc2)')
    parser.add_argument('--output_file', type=pathlib.Path, required=True,
                        help='Path to desired output file (.png for stills or .mkv for animations)')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--render_still', type=int,
                       help='Optional number [frame] to render a still')
    group.add_argument('--render_animation', type=int, nargs=3,
                       help='Optional list of numbers [frame_start, frame_end, frames_per_second] to render an animation')

    parser.add_argument('--num_samples', type=int,
                       help='Optional number of render samples')
    
    args = parser.parse_args()

    render(**vars(args))