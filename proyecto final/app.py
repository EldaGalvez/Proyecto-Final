import pandas as pd
import plotly.express as px
from statsmodels.tsa.statespace.sarimax import SARIMAX
from shiny import App, ui, render, reactive
from shinywidgets import render_plotly
import faicons as fa

# Cargar datos
df = pd.read_csv("c:/Users/Eldita/Documents/8vo semestre/Big Data/proyecto final/Aforos-RedPropia.csv", encoding="latin-1")

vehiculos = [
    "AUTOS", "MOTOS", "AUTOBUS DE 2 EJES", "AUTOBUS DE 3 EJES", "AUTOBUS DE 4 EJES",
    "CAMIONES DE 2 EJES", "CAMIONES DE 3 EJES", "CAMIONES DE 4 EJES", "CAMIONES DE 5 EJES",
    "CAMIONES DE 6 EJES", "CAMIONES DE 7 EJES", "CAMIONES DE 8 EJES", "CAMIONES DE 9 EJES",
    "TRICICLOS", "EJE EXTRA AUTOBUS", "EJE EXTRA CAMION", "PEATONES"
]

for v in vehiculos:
    df[v] = df[v].astype(str).str.replace(",", "", regex=False).str.strip()
    df[v] = pd.to_numeric(df[v], errors="coerce")

df["MES"] = df["MES"].str.upper()
meses_dict = {
    "ENERO": 1, "FEBRERO": 2, "MARZO": 3, "ABRIL": 4,
    "MAYO": 5, "JUNIO": 6, "JULIO": 7, "AGOSTO": 8,
    "SEPTIEMBRE": 9, "OCTUBRE": 10, "NOVIEMBRE": 11, "DICIEMBRE": 12
}
df["MES_NUM"] = df["MES"].map(meses_dict)
df["FECHA"] = pd.to_datetime(df["AÑO"].astype(str) + "-" + df["MES_NUM"].astype(str) + "-01")

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_slider("rango_anios", "Período de tiempo", 2021, 2025, value=(2021, 2025)),
        ui.input_select("vehiculo", "Tipo de vehículo", vehiculos, selected="AUTOS"),
        ui.input_numeric("mes", "Mes seleccionado (1-12)", value=6, min=1, max=12),
        ui.input_numeric("anio", "Año seleccionado", value=2025, min=2021, max=2025),
        ui.input_checkbox_group("vehiculos_check", "Mostrar vehículos", vehiculos, inline=True),
    ),
    ui.h2("Movimientos mensuales por tipo de vehículo en la red CAPUFE"),
    ui.layout_columns(
        ui.value_box(ui.output_text("titulo_total"), ui.output_text("total_anual"), showcase=ui.output_ui("icono_dinamico")),
        ui.value_box("Frecuencia", ui.output_text("frecuencia"), showcase=fa.icon_svg("calendar")),
        ui.value_box("Pronóstico mensual", ui.output_text("forecast_box"), showcase=fa.icon_svg("chart-line")),
    ),
    ui.layout_columns(
        ui.card(ui.card_header("Conjunto de datos"), ui.output_data_frame("tabla")),
        ui.card(ui.card_header("Pronóstico (Mes)"), ui.output_plot("grafico_plotly")),
    ),
    ui.layout_columns(
        ui.card(ui.card_header("Frecuencias de tipos de vehículo por año"), ui.output_plot("grafico_barras")),
        ui.card(ui.card_header("Cantidad de vehículos"), ui.output_ui("conteo_vehiculos")),
        ui.card(ui.card_header("Estadísticas:"), ui.output_ui("estadisticas"))
    )
)

def server(input, output, session):

    @reactive.Calc
    def resumen():
        tipo = input.vehiculo()
        anio = input.anio()
        mes = input.mes()
        rango = input.rango_anios()

        serie = df.groupby("FECHA")[tipo].sum().asfreq("MS").ffill()
        serie = serie[serie.index.year >= 2021]

        valor_predicho = 0
        forecast = pd.Series()

        if not serie.empty and serie.sum() > 0:
            n = int(len(serie) * 0.8)
            train = serie.iloc[:n]
            try:
                modelo = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                                 enforce_stationarity=False, enforce_invertibility=False)
                resultado = modelo.fit(disp=False)
                future_steps = pd.date_range(start=serie.index[n], periods=36, freq="MS")
                forecast = resultado.get_forecast(steps=len(future_steps)).predicted_mean
                fecha_pred = pd.to_datetime(f"{anio}-{mes:02d}-01")
                if fecha_pred in forecast.index:
                    valor_predicho = int(forecast.get(fecha_pred, 0))
            except:
                valor_predicho = 0

        total = df[(df["AÑO"] >= rango[0]) & (df["AÑO"] <= rango[1])][tipo].sum()
        seleccionados = list(input.vehiculos_check())
        tabla = df[(df["AÑO"] == anio) & (df["MES_NUM"] == mes)].sort_values("FECHA")[["AÑO", "MES"] + seleccionados]

        return total, valor_predicho, tabla, serie, forecast

    @output
    @render.text
    def titulo_total():
        tipo = input.vehiculo().title()
        hasta = input.rango_anios()[1]
        return f"Total de {tipo} hasta {hasta:,}"

    @output
    @render.text
    def total_anual():
        return f"{resumen()[0]:,}"

    @output
    @render.text
    def forecast_box():
        return f"{resumen()[1]:,}"

    @output
    @render.text
    def frecuencia():
        tipo = input.vehiculo()
        a1, a2 = input.rango_anios()
        datos = df[(df["AÑO"] >= a1) & (df["AÑO"] <= a2)]
        serie = datos.groupby("FECHA")[tipo].sum().sort_index()
        if serie.empty:
            return "Sin datos"
        frecuencia = pd.infer_freq(serie.index)
        if frecuencia == "MS":
            return "Mensual"
        elif frecuencia == "QS":
            return "Trimestral"
        elif frecuencia == "AS":
            return "Anual"
        else:
            return "Indefinida"

    @output
    @render.ui
    def icono_dinamico():
        iconos = {
            "AUTOS": "car", "MOTOS": "motorcycle", "AUTOBUS DE 2 EJES": "bus",
            "AUTOBUS DE 3 EJES": "bus", "AUTOBUS DE 4 EJES": "bus",
            "CAMIONES DE 2 EJES": "truck", "CAMIONES DE 3 EJES": "truck",
            "CAMIONES DE 4 EJES": "truck", "CAMIONES DE 5 EJES": "truck",
            "CAMIONES DE 6 EJES": "truck", "CAMIONES DE 7 EJES": "truck",
            "CAMIONES DE 8 EJES": "truck", "CAMIONES DE 9 EJES": "truck",
            "TRICICLOS": "bicycle", "EJE EXTRA AUTOBUS": "plus",
            "EJE EXTRA CAMION": "plus", "PEATONES": "person-walking"
        }
        icono = iconos.get(input.vehiculo(), "car")
        return fa.icon_svg(icono)

    @output
    @render.data_frame
    def tabla():
        return resumen()[2]

    @output
    @render_plotly
    def grafico_plotly():
        serie = resumen()[3]
        forecast = resumen()[4]

        if serie.empty or serie.sum() == 0:
            return px.line(title="Sin datos para graficar")

        fig = px.line()
        fig.add_scatter(x=serie.index, y=serie, mode="lines+markers", name="Datos reales", line=dict(color="blue"))

        if not forecast.empty:
            fig.add_scatter(x=forecast.index, y=forecast, mode="lines+markers", name="Pronóstico", line=dict(color="red"))

        fig.update_layout(
            title="Pronóstico mensual con SARIMAX",
            xaxis_title="Fecha",
            yaxis_title="Cantidad de vehículos",
            legend_title=""
        )
        return fig

    @output
    @render_plotly
    def grafico_barras():
        a1, a2 = input.rango_anios()
        seleccionados = list(input.vehiculos_check())
        df_filtrado = df[(df["AÑO"] >= a1) & (df["AÑO"] <= a2)]

        if df_filtrado.empty or not seleccionados:
            return px.line(title="Sin datos para graficar")

        df_group = df_filtrado.groupby("AÑO")[seleccionados].sum().reset_index()
        df_long = pd.melt(df_group, id_vars="AÑO", var_name="Tipo de Vehículo", value_name="Cantidad")

        fig = px.bar(df_long, x="AÑO", y="Cantidad", color="Tipo de Vehículo", barmode="group")
        fig.update_layout(
            title="Frecuencias de vehículos por año",
            xaxis_title="Año",
            yaxis_title="Cantidad",
            legend_title="Tipo de vehículo"
        )
        return fig

    @output
    @render.ui
    def conteo_vehiculos():
        iconos = {v: "•" for v in vehiculos}
        seleccionados = list(input.vehiculos_check())
        totales = df[df["AÑO"] == input.anio()][seleccionados].sum().astype(int)
        html = ""
        for tipo in seleccionados:
            icono = iconos.get(tipo, "•")
            cantidad = f"{totales[tipo]:,}"
            html += f"<p>{icono} <strong>{tipo.title()}</strong>: {cantidad}</p>"

        return ui.HTML(html)

    @output
    @render.ui
    def estadisticas():
        seleccionados = list(input.vehiculos_check())
        suma_tipos = df[df["AÑO"] == input.anio()][seleccionados].sum()

        if suma_tipos.empty or suma_tipos.isnull().all():
            return ui.HTML("⚠️ No hay datos suficientes para mostrar estadísticas.")

        mayor = suma_tipos.idxmax()
        menor = suma_tipos.idxmin()

        return ui.HTML(
            f"Tipo de vehículo con mayor movimiento:<br><strong>{mayor}</strong><br><br>"
            f"Tipo con menor movimiento:<br><strong>{menor}</strong>"
        )

app = App(app_ui, server)
