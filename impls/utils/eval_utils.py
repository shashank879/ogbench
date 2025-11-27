import numpy as np
import cv2
import wandb


def visualize_goal_buffer_on_maze(
    goal_buffer, 
    env, 
    render_size=1024, 
    goal_color=(0, 255, 0), 
    goal_radius=4,
    value_fn=None,
    arrow_scale=2,
    arrow_color=(255, 0, 0),
    arrow_thickness=1,
    arrow_norm_stat='percentile',
):
    """Visualize all stored goals in the buffer on a rendered maze image.

    Args:
        goal_buffer: GoalBuffer instance containing stored goals
        env: The maze environment (ogbench MazeEnv)
        render_size: Size of the rendered image (width=height)
        goal_color: RGB color for goal markers (default: green)
        goal_radius: Radius of goal markers in pixels
        value_fn: Optional value function v(s,g) that takes batched (states, goals) 
                  and returns values. If provided, visualizes value gradients.
        arrow_scale: Scale factor for gradient arrows (in pixels)
        arrow_color: RGB color for gradient arrows (default: red)
        arrow_thickness: Thickness of gradient arrow lines

    Returns:
        rendered_image: RGB image (H, W, 3) with goals and gradients visualized
    """
    import jax
    import jax.numpy as jnp

    # Render the maze
    _, info = env.reset()
    env_goal = info.get('goal')
    frame = env.render()
    frame = cv2.resize(frame, (render_size, render_size))

    if len(goal_buffer.goal_observations) == 0:
        # Return empty render if buffer is empty
        return frame

    # Get maze parameters from environment
    maze_map = env.unwrapped.maze_map if hasattr(env, 'unwrapped') else env.maze_map
    maze_type = env.unwrapped._maze_type if hasattr(env, 'unwrapped') else env._maze_type

    # Extract xy coordinates from all goal observations
    goal_xys = []
    for goal_obs in goal_buffer.goal_observations:
        # Convert from JAX array to numpy if needed
        goal_obs_np = np.array(goal_obs)
        goal_xys.append(goal_obs_np[:2])
    goal_xys = np.array(goal_xys)  # Shape: (num_goals, 2)

    # Convert xy coordinates to pixel coordinates
    goal_pixels = xy_to_pixel_coords(
        goal_xys, 
        maze_map=maze_map,
        maze_type=maze_type,
        render_size=frame.shape[0]  # Assuming square render
    )

    # Draw goals on the frame
    frame_with_goals = frame.copy()
    for pixel_x, pixel_y in goal_pixels:
        # Ensure coordinates are within bounds
        if 0 <= pixel_x < frame.shape[1] and 0 <= pixel_y < frame.shape[0]:
            cv2.circle(
                frame_with_goals, 
                (int(pixel_x), int(pixel_y)), 
                goal_radius, 
                goal_color, 
                -1  # Filled circle
            )

    # Add final goal
    env_goal_pixels = xy_to_pixel_coords(
        env_goal[None], 
        maze_map=maze_map,
        maze_type=maze_type,
        render_size=frame.shape[0]  # Assuming square render
    )[0]
    cv2.circle(
        frame_with_goals, 
        (int(env_goal_pixels[0]), int(env_goal_pixels[1])), 
        goal_radius * 2,
        (0,0,0), 
        -1  # Filled circle
    )

    # Compute and visualize value gradients if value_fn is provided
    if value_fn is not None and env_goal is not None:
        # Convert buffer observations to JAX arrays for gradient computation
        buffer_states = jnp.array([np.array(obs) for obs in goal_buffer.goal_observations])

        # Expand goal to match batch size
        goal_batch = jnp.tile(jnp.array(env_goal)[None, :], (len(buffer_states), 1))

        # Define function to compute gradients
        def compute_value_gradient(states, goals):
            """Compute gradient of value function wrt states."""
            def value_for_state(state):
                # Compute value for single state with given goal
                v = value_fn(state[None, :], goals[0:1])[0]
                return v

            # Compute gradients for each state
            grad_fn = jax.grad(value_for_state)
            gradients = jax.vmap(grad_fn)(states)
            return gradients

        # Compute gradients
        gradients = compute_value_gradient(buffer_states, goal_batch)
        gradients_np = np.array(gradients)  # Shape: (num_states, state_dim)

        # Extract only x,y components (first 2 dimensions)
        gradients_xy = gradients_np[:, :2]  # Shape: (num_states, 2)

        # Compute magnitudes for normalization
        magnitudes = np.linalg.norm(gradients_xy, axis=1, keepdims=True)
        magnitudes = np.maximum(magnitudes, 1e-8)  # Avoid division by zero

        # Normalize gradients and scale
        normalized_gradients = gradients_xy / magnitudes
        scaled_gradients = normalized_gradients * arrow_scale

        # Also scale by relative magnitude for proportional sizing
        norm_stat = {'median': np.median, 'mean': np.mean, 'max': np.max, 'percentile': (lambda x: np.percentile(x, 95))}[arrow_norm_stat](magnitudes)
        if norm_stat > 0:
            magnitude_scale = magnitudes / norm_stat
            # magnitude_scale = np.maximum(magnitude_scale, 0.15)
            scaled_gradients = scaled_gradients * magnitude_scale

        # Convert gradient endpoints to pixel coordinates
        # Start points are the goal positions
        start_xys = goal_xys  # Shape: (num_goals, 2)
        end_xys = start_xys + scaled_gradients  # Shape: (num_goals, 2)

        # Convert to pixel coordinates
        start_pixels = xy_to_pixel_coords(
            start_xys,
            maze_map=maze_map,
            maze_type=maze_type,
            render_size=frame.shape[0]
        )

        end_pixels = xy_to_pixel_coords(
            end_xys,
            maze_map=maze_map,
            maze_type=maze_type,
            render_size=frame.shape[0]
        )

        # Draw gradient arrows
        for (start_x, start_y), (end_x, end_y) in zip(start_pixels, end_pixels):
            if (0 <= start_x < frame.shape[1] and 0 <= start_y < frame.shape[0] and
                0 <= end_x < frame.shape[1] and 0 <= end_y < frame.shape[0]):
                cv2.line(
                    frame_with_goals,
                    (int(start_x), int(start_y)),
                    (int(end_x), int(end_y)),
                    arrow_color,
                    arrow_thickness,
                    cv2.LINE_AA
                )

    return frame_with_goals


def xy_to_pixel_coords(xys, maze_map, maze_type, render_size):
    """Convert maze xy coordinates to pixel coordinates in rendered image.

    Args:
        xys: Array of shape (N, 2) containing xy coordinates
        maze_map: The maze map array
        maze_type: Type of maze ('arena', 'medium', 'large', 'giant', 'teleport')
        render_size: Size of the rendered image (assumes square)

    Returns:
        pixel_coords: Array of shape (N, 2) containing pixel coordinates
    """
    maze_height = maze_map.shape[0]
    maze_width = maze_map.shape[1]

    # Maze rendering parameters (from ogbench camera setup)
    # Camera looks at center of navigable space
    center_x = 2 * (maze_width - 3)
    center_y = 2 * (maze_height - 3)

    # Camera distance determines the view frustum
    # distance = 5 * (maze_width - 2)

    # The maze bounds in world coordinates
    maze_unit = 4.0
    offset_x = 4.0
    offset_y = 4.0

    # World coordinate bounds
    world_min_x = 0 * maze_unit - offset_x
    world_max_x = (maze_width - 1) * maze_unit - offset_x
    world_min_y = 0 * maze_unit - offset_y
    world_max_y = (maze_height - 1) * maze_unit - offset_y

    # Normalize xy to [0, 1] range based on world bounds
    # Note: y-axis is typically flipped in image coordinates
    normalized_x = (xys[:, 0] - world_min_x) / (world_max_x - world_min_x)
    normalized_y = (xys[:, 1] - world_min_y) / (world_max_y - world_min_y)

    # Convert to pixel coordinates
    # Y is flipped (top of image is y=0)
    pixel_x = normalized_x * render_size
    pixel_y = (1 - normalized_y) * render_size  # Flip y-axis

    pixel_coords = np.stack([pixel_x, pixel_y], axis=1)
    return pixel_coords


def is_visual_observation(obs):
    """Check if observation is visual (image) or state-based (vector)."""
    obs_array = np.array(obs)
    return len(obs_array.shape) >= 3  # Images have at least 3 dimensions (H, W, C)


def visualize_goals_on_trajectory(
    trajectories,
    env,
    renders=None,
    render_size=400,
    decoded_goal_color=(255, 0, 0),  # Red for decoded/predicted goals
    retrieved_goal_color=(0, 0, 255),  # Blue for retrieved goals
    final_goal_color=(255, 255, 0),  # Yellow for final goal
    goal_radius=2,
    goal_buffer=None,
    show_indices=True,
    font_scale=0.4,
    font_thickness=1,
    goal_layout='horizontal',  # 'horizontal', 'vertical', or 'grid'
    max_goals_display=10,
    resize_goals=True,
    goal_border_width=2,
):
    """Visualize goals on trajectory renders for both state-based and visual observations.

    Automatically detects whether subgoals are state-based (coordinates) or visual (images)
    and uses the appropriate visualization method.

    Creates renders showing:
    - For state-based: Colored circles at goal positions
    - For visual: Concatenated subgoal images with colored borders

    Args:
        trajectories: List of trajectory dicts from evaluate()
        env: The environment
        renders: Pre-rendered frames for each trajectory
        render_size: Size of rendered frames

        # State-based visualization parameters:
        decoded_goal_color: RGB color for decoded goals
        retrieved_goal_color: RGB color for retrieved goals
        final_goal_color: RGB color for final target goal
        goal_radius: Radius of goal markers in pixels
        goal_buffer: Optional buffer of goals to visualize

        # Visual observation parameters:
        goal_layout: How to arrange multiple subgoals ('horizontal', 'vertical', 'grid')
        max_goals_display: Maximum number of subgoals to show
        resize_goals: Whether to resize subgoal images to match frame dimensions
        goal_border_width: Width of borders between goals

        # Common parameters:
        show_indices: Whether to show index numbers
        font_scale: Size of index text
        font_thickness: Thickness of index text

    Returns:
        List of frame arrays with goal visualizations
    """
    # Detect if we're using visual observations
    use_visual = False
    for traj in trajectories:
        if 'decoded_subgoals' in traj and len(traj['decoded_subgoals']) > 0:
            if len(traj['decoded_subgoals'][0]) > 0:
                use_visual = is_visual_observation(traj['decoded_subgoals'][0][0])
                break
        elif 'subgoals' in traj and len(traj['subgoals']) > 0:
            if len(traj['subgoals'][0]) > 0:
                use_visual = is_visual_observation(traj['subgoals'][0][0])
                break

    if use_visual:
        return _visualize_visual_goals(
            trajectories, env, renders, render_size,
            goal_layout, max_goals_display, resize_goals,
            decoded_goal_color, retrieved_goal_color, final_goal_color,
            goal_border_width, show_indices, font_scale, font_thickness
        )
    else:
        return _visualize_state_goals(
            trajectories, env, renders, render_size,
            decoded_goal_color, retrieved_goal_color, final_goal_color,
            goal_radius, goal_buffer, show_indices, font_scale, font_thickness
        )


def _visualize_state_goals(
    trajectories, env, renders, render_size,
    decoded_goal_color, retrieved_goal_color, final_goal_color,
    goal_radius, goal_buffer, show_indices, font_scale, font_thickness
):
    """Visualize state-based goals as circles on trajectory (original implementation)."""

    maze_map = env.unwrapped.maze_map if hasattr(env, 'unwrapped') else env.maze_map
    maze_type = env.unwrapped._maze_type if hasattr(env, 'unwrapped') else env._maze_type
    all_frames = []

    for traj, render in zip(trajectories, renders):
        if renders:
            frame_size = render.shape[2]
            assert render.shape[1] in [frame_size, 2*frame_size], f'Frame height in shape ({render.shape}) must in [{frame_size},{2*frame_size}]'
        else:
            frame_size = render_size
        assert len(traj['observation']) == len(render), 'Length mismatch'

        # Get final goal from trajectory info
        final_goal_obs = traj['goal'][-1]
        if final_goal_obs is not None:
            final_goal_xy = np.array(final_goal_obs)[:2]
        else:
            final_goal_xy = None

        traj_frames = []
        for step_idx in range(len(traj['observation'])):
            obs = traj['observation'][step_idx]
            subgoal_first_reach_index = traj['subgoal_first_reach_index'][step_idx] if 'subgoal_first_reach_index' in traj else -1

            agent_xy = obs[:2]
            if renders:
                render_frame = render[min(step_idx, len(render) - 1)]
            else:
                env.unwrapped.set_xy(agent_xy) if hasattr(env, 'unwrapped') else env.set_xy(agent_xy)
                render_frame = env.render().copy()

            frame = np.zeros((frame_size, frame_size, 3), dtype=np.uint8) + 255

            if goal_buffer:
                # Extract xy coordinates from all goal observations
                # Assuming goal_observations have xy in first 2 dimensions
                goal_xys = []
                for goal_obs in goal_buffer.goal_observations:
                    # Convert from JAX array to numpy if needed
                    goal_obs_np = np.array(goal_obs)
                    goal_xys.append(goal_obs_np[:2])
                goal_xys = np.array(goal_xys)  # Shape: (num_goals, 2)

                # Convert xy coordinates to pixel coordinates
                goal_pixels = xy_to_pixel_coords(
                    goal_xys, 
                    maze_map=maze_map,
                    maze_type=maze_type,
                    render_size=frame_size  # Assuming square render
                )

                # Draw goals on the frame
                for pixel_x, pixel_y in goal_pixels:
                    if frame.shape[0] == 2*frame_size:
                        pixel_y += frame_size
                    if 0 <= pixel_x < frame.shape[1] and 0 <= pixel_y < frame.shape[0]:
                        cv2.circle(frame, (int(pixel_x), int(pixel_y)), 1, (0, 255, 0), -1)

            goals_to_draw = []

            # Add decoded goals
            if 'decoded_subgoals' in traj and step_idx < len(traj['decoded_subgoals']):
                decoded_subgoals = traj['decoded_subgoals'][step_idx]
                if subgoal_first_reach_index > -1:
                    decoded_subgoals = decoded_subgoals[:subgoal_first_reach_index+1]

                for idx, decoded_goal in enumerate(decoded_subgoals):
                    decoded_goal_xy = np.array(decoded_goal)[:2]
                    goals_to_draw.append((decoded_goal_xy, decoded_goal_color, 'decoded', 'G' if idx==0 else (idx-1)))

            # Add retrieved goals
            if 'subgoals' in traj and step_idx < len(traj['subgoals']):
                retrieved_subgoals = traj['subgoals'][step_idx]
                if subgoal_first_reach_index > -1:
                    retrieved_subgoals = retrieved_subgoals[:subgoal_first_reach_index+1]
                for idx, retrieved_subgoal in enumerate(retrieved_subgoals):
                    retrieved_goal_xy = np.array(retrieved_subgoal)[:2]
                    goals_to_draw.append((retrieved_goal_xy, retrieved_goal_color, 'retrieved',  'G' if idx==0 else (idx-1)))

            # Add final target goal
            if final_goal_xy is not None:
                goals_to_draw.append((final_goal_xy, final_goal_color, 'target', None))

            goals_to_draw.append((agent_xy, (0, 0, 0), 'agent', None))

            # Draw goals on frame
            if len(goals_to_draw) > 0:
                goal_xys = np.array([g[0] for g in goals_to_draw])
                goal_pixels = xy_to_pixel_coords(
                    goal_xys,
                    maze_map=maze_map,
                    maze_type=maze_type,
                    render_size=frame_size,
                )

                for (pixel_x, pixel_y), goal_data in zip(goal_pixels, goals_to_draw):
                    _, color, goal_type, idx = goal_data
                    if 0 <= pixel_x < frame_size and 0 <= pixel_y < frame_size:
                        adjusted_pixel_y = pixel_y
                        if frame.shape[0] == frame_size * 2:
                            adjusted_pixel_y += frame_size

                        cv2.circle(
                            frame,
                            (int(pixel_x), int(adjusted_pixel_y)),
                            goal_radius,
                            color,
                            -1
                        )
                        # Draw index number if enabled and available
                        if show_indices and idx is not None and goal_type in ['decoded', 'retrieved']:
                            # Position text slightly offset from the circle
                            text_x = int(pixel_x) + goal_radius + 3
                            text_y = int(adjusted_pixel_y) - goal_radius - 3

                            # Draw text with background for better visibility
                            text = str(idx)
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]

                            # Draw text
                            cv2.putText(
                                frame,
                                text,
                                (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                font_scale,
                                color,
                                font_thickness,
                                cv2.LINE_AA
                            )

            traj_frames.append(np.concatenate([render_frame, frame], 0))
        all_frames.append(np.array(traj_frames))

    return all_frames


def _visualize_visual_goals(
    trajectories, env, renders, render_size,
    goal_layout, max_goals_display, resize_goals,
    decoded_goal_color, retrieved_goal_color, final_goal_color,
    goal_border_width, show_indices, font_scale, font_thickness,
    upscale_factor=4.0,
):
    """Visualize visual goals as concatenated images."""
    all_frames = []

    for traj_idx, (traj, render) in enumerate(zip(trajectories, renders)):
        if renders is not None:
            frame_height, frame_width = render.shape[1], render.shape[2]
        else:
            frame_height = frame_width = render_size

        assert len(traj['observation']) == len(render), f'Length mismatch: {len(traj["observation"])} vs {len(render)}'

        traj_frames = []

        for step_idx in range(len(traj['observation'])):
            # Get the render frame
            if renders is not None:
                render_frame = render[min(step_idx, len(render) - 1)].copy()
            else:
                obs = traj['observation'][step_idx]
                agent_xy = obs[:2]
                env.unwrapped.set_xy(agent_xy) if hasattr(env, 'unwrapped') else env.set_xy(agent_xy)
                render_frame = env.render().copy()

            # Collect subgoals to visualize
            subgoals_to_viz = []
            subgoal_first_reach_index = traj.get('subgoal_first_reach_index', [None])[step_idx]

            # Check for retrieved subgoals
            if 'subgoals' in traj and step_idx < len(traj['subgoals']):
                retrieved_subgoals = traj['subgoals'][step_idx]
                if subgoal_first_reach_index is not None and subgoal_first_reach_index > -1:
                    retrieved_subgoals = retrieved_subgoals[:subgoal_first_reach_index + 1]

                for idx, subgoal in enumerate(retrieved_subgoals[:max_goals_display]):
                    subgoals_to_viz.append({
                        'image': np.array(subgoal),
                        'type': 'retrieved',
                        'index': 'G' if idx == 0 else str(idx - 1),
                        'border_color': final_goal_color if idx==0 else retrieved_goal_color
                    })

            # # Add final goal
            # if 'goal' in traj and step_idx < len(traj['goal']):
            #     final_goal = traj['goal'][step_idx]
            #     if final_goal is not None and is_visual_observation(final_goal):
            #         subgoals_to_viz.append({
            #             'image': np.array(final_goal),
            #             'type': 'final',
            #             'index': 'F',
            #             'border_color': final_goal_color
            #         })

            # Create visualization of subgoals
            if len(subgoals_to_viz) > 0:
                goal_viz = _create_goal_visualization(
                    list(reversed(subgoals_to_viz)),
                    target_height=int(frame_height / 4 * upscale_factor),
                    target_width=int(frame_width * upscale_factor),
                    layout=goal_layout,
                    border_width=goal_border_width,
                    show_indices=show_indices,
                    font_scale=font_scale,
                    font_thickness=font_thickness,
                    resize_goals=resize_goals
                )

                # Concatenate render and goal visualization
                if upscale_factor != 1:
                    render_frame = cv2.resize(render_frame, (int(render_frame.shape[1] * upscale_factor), int(render_frame.shape[0] * upscale_factor)))
                combined_frame = np.concatenate([render_frame, goal_viz], axis=0)
            else:
                # No subgoals, just use render
                combined_frame = render_frame

            traj_frames.append(combined_frame)

        all_frames.append(np.array(traj_frames))

    return all_frames


def _create_goal_visualization(
    subgoals,
    target_height,
    target_width,
    layout='horizontal',
    border_width=2,
    show_indices=True,
    font_scale=0.5,
    font_thickness=2,
    resize_goals=True
):
    """Create a visualization panel for subgoal images."""
    if len(subgoals) == 0:
        return np.ones((target_height, target_width, 3), dtype=np.uint8) * 255

    # Process each subgoal image
    processed_goals = []
    for sg in subgoals:
        img = sg['image'].copy()

        # Ensure image is uint8 and has 3 channels
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Add colored border
        border_c = sg['border_color']
        img = cv2.copyMakeBorder(
            img, 
            border_width, border_width, border_width, border_width,
            cv2.BORDER_CONSTANT, 
            value=border_c
        )

        # Add index label if requested
        if show_indices and sg['index'] is not None:
            text = str(sg['index'])
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
            cv2.rectangle(img, (5, 5), (15 + text_size[0], 15 + text_size[1]), (0, 0, 0), -1)
            cv2.putText(
                img, text, (10, 10 + text_size[1]),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                font_thickness, cv2.LINE_AA
            )

        processed_goals.append(img)

    # Arrange based on layout
    if layout == 'horizontal':
        combined = np.concatenate(processed_goals, axis=1)
    elif layout == 'vertical':
        combined = np.concatenate(processed_goals, axis=0)
    elif layout == 'grid':
        n_goals = len(processed_goals)
        n_cols = int(np.ceil(np.sqrt(n_goals)))
        n_rows = int(np.ceil(n_goals / n_cols))

        # Pad to fill grid
        while len(processed_goals) < n_rows * n_cols:
            processed_goals.append(np.ones_like(processed_goals[0]) * 255)

        rows = []
        for r in range(n_rows):
            row = np.concatenate(processed_goals[r * n_cols:(r + 1) * n_cols], axis=1)
            rows.append(row)
        combined = np.concatenate(rows, axis=0)
    else:
        raise NotImplementedError(layout)

    # Resize to target dimensions if requested
    if resize_goals:
        h, w = combined.shape[:2]
        # Calculate scaling to fit within target while preserving aspect ratio
        scale = min(target_width / w, target_height / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        combined = cv2.resize(combined, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to exactly match target dimensions
        pad_h = target_height - new_h
        pad_w = target_width - new_w
        combined = cv2.copyMakeBorder(
            combined, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
    else:
        # Pad or crop to match target dimensions
        h, w = combined.shape[:2]
        if h < target_height or w < target_width:
            pad_h = max(0, target_height - h)
            pad_w = max(0, target_width - w)
            combined = cv2.copyMakeBorder(
                combined, 0, pad_h, 0, pad_w,
                cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
        if combined.shape[0] > target_height or combined.shape[1] > target_width:
            combined = combined[:target_height, :target_width]

    return combined

def create_goal_trajectory_video(
    trajectories,
    env,
    renders=None,
    render_size=400,
    n_cols=None,
    goal_buffer=None,
    **kwargs
):
    """Create a video grid showing goal visualizations across trajectories.

    Works for both state-based and visual observations.

    Args:
        trajectories: List of trajectory dicts from evaluate()
        env: The maze environment
        renders: Pre-rendered frames
        render_size: Size of rendered frames
        n_cols: Number of columns in video grid (auto if None)
        goal_buffer: Optional buffer of goals (for state-based only)
        **kwargs: Additional arguments passed to visualize_goals_on_trajectory

    Returns:
        wandb.Video object
    """
    # Generate frames with goal visualizations
    trajectory_frames = visualize_goals_on_trajectory(
        trajectories, env, renders=renders, render_size=render_size,
        goal_buffer=goal_buffer, **kwargs
    )

    # Pad trajectories to same length
    max_len = max(len(frames) for frames in trajectory_frames)
    padded_frames = []

    for frames in trajectory_frames:
        if len(frames) < max_len:
            last_frame = frames[-1]
            padding = np.stack([last_frame] * (max_len - len(frames)))
            frames = np.concatenate([frames, padding], axis=0)
        padded_frames.append(frames)

    # Stack into grid
    if n_cols is None:
        n_cols = int(np.ceil(np.sqrt(len(padded_frames))))
    n_rows = int(np.ceil(len(padded_frames) / n_cols))

    # Pad to fill grid
    while len(padded_frames) < n_rows * n_cols:
        padded_frames.append(np.zeros_like(padded_frames[0]))

    # Reshape into grid
    grid_frames = []
    for t in range(max_len):
        frame_grid = []
        for row in range(n_rows):
            row_frames = []
            for col in range(n_cols):
                idx = row * n_cols + col
                if idx < len(padded_frames):
                    row_frames.append(padded_frames[idx][t])
            frame_grid.append(np.concatenate(row_frames, axis=1))
        grid_frame = np.concatenate(frame_grid, axis=0)
        grid_frames.append(grid_frame)

    video_array = np.array(grid_frames)
    # Convert to format expected by wandb (T, H, W, C) -> (T, C, H, W)
    video_array = video_array.transpose(0, 3, 1, 2)

    return wandb.Video(video_array, fps=20, format="mp4")
