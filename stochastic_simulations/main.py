import random
import simulations as sim
import numpy as np
import streamlit as st
import plotly.graph_objects as go
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    st.title("Stochastic Processes Analysis")

    st.set_page_config(layout="wide", page_title="SDE Simulator")
    st.title("Universal SDE Simulator")
    st.markdown("Define any stochastic process with custom drift and diffusion functions")

### SIDEBAR
    with st.sidebar:
        st.header("Process Definition")

        # Drift function input
        st.subheader("Drift Function: a(t, x)")
        drift_input = st.text_area(
            "Define drift (use 't' for time, 'x' for value)",
            value="0.05 * x",
            height=100,
            help="Example: 0.05*x (GBM), 1.0*(0.5-x) (Ornstein-Uhlenbeck)"
        )

        # Diffusion function input
        st.subheader("Diffusion Function: b(t, x)")
        diffusion_input = st.text_area(
            "Define diffusion (use 't' for time, 'x' for value)",
            value="0.2 * x",
            height=100,
            help="Example: 0.2*x (GBM), 0.1 (Ornstein-Uhlenbeck)"
        )

        st.divider()
        st.subheader("Simulation Parameters")

        # Time parameters
        t_start = st.number_input("Start Time", value=0.0, step=0.1)
        t_end = st.number_input("End Time", value=1.0, step=0.1)

        # Process parameters
        n_steps = st.slider("Number of Time Steps", 50, 1000, 252)
        n_paths = st.slider("Number of Paths", 1, 100, 10)
        x0 = st.number_input("Initial Value (x₀)", value=100.0, step=1.0)

        # Numerical scheme
        use_milstein = st.checkbox("Use Milstein Method", value=False)

        st.divider()

        # Preset examples
        st.subheader("Preset Examples")
        preset = st.selectbox(
            "Load Preset",
            ["None", "GBM (μ=5%, σ=20%)", "Ornstein-Uhlenbeck", "Square Root (CIR)"]
        )

        if preset == "GBM (μ=5%, σ=20%)":
            drift_input = "0.05 * x"
            diffusion_input = "0.2 * x"
            st.success("✓ GBM loaded")
        elif preset == "Ornstein-Uhlenbeck":
            drift_input = "1.0 * (0.5 - x)"
            diffusion_input = "0.1"
            st.success("✓ OU loaded")
        elif preset == "Square Root (CIR)":
            drift_input = "0.3 * (0.05 - x)"
            diffusion_input = "0.1 * np.sqrt(x)"
            st.success("✓ CIR loaded")

    try:
        drift_func = sim.parse_function(drift_input)
        diffusion_func = sim.parse_function(diffusion_input)

        # params = st.session_state.get('sim_params', {})

        st.info(f"""
            **Simulation Parameters:**
            - Time: [{t_start} to {t_end}]
            - Steps: {n_steps}
            - Paths: {n_paths}
            - Initial Value: {x0}
            - Milstein: {use_milstein}
            """)

        process_euler, fig_euler = sim.simulation(a=drift_func, b=diffusion_func, time=[t_start, t_end], n=n_steps, m=n_paths, x0=x0, plot=True)
        st.plotly_chart(fig_euler)

        process_euler, fig_milstein = sim.simulation(a=drift_func, b=diffusion_func, time=[t_start, t_end], n=n_steps,
                                                  m=n_paths, x0=x0, plot=True, milstein=True)
        st.plotly_chart(fig_milstein)

        process_euler, fig_pred_corr = sim.simulation(a=drift_func, b=diffusion_func, time=[t_start, t_end], n=n_steps,
                                                  m=n_paths, x0=x0, plot=True, predictor_corrector=True)
        st.plotly_chart(fig_pred_corr)
        # st.write(f"process: {process}")
    except ValueError as e:
        st.error(f"❌ Cannot simulate: {str(e)}")

    # a = np.linspace(0, 100)
    # b = a*a
    #
    # fig = go.Figure()
    #
    # fig.add_trace(go.Scatter(
    #     x=a,
    #     y=b,
    #     mode='lines',
    #     name=f'Path'
    # ))
    #
    # fig.update_layout(
    #     title="<b>Simulated Stochastic Paths</b>",
    #     xaxis_title="Time",
    #     yaxis_title="Values",
    #     hovermode='x unified',
    #     template='plotly_white'
    # )
    #
    # # fig.show()
    # st.plotly_chart(fig)
