window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
    refresh_graph: function (
      i,
      x_click,
      y_click,
      click,
      close_modal,
      x,
      y,
      z,
      fig,
      y_scatter,
      yz_scatter,
      x_scatter,
      xz_scatter,
      line_switch,
      x_title,
      y_title
    ) {
      const triggered_id = dash_clientside.callback_context.triggered.map(
        (t) => t.prop_id
      )[0];

      if (fig === undefined) {
        return window.dash_clientside.no_update;
      }
      figure = JSON.parse(JSON.stringify(fig));

      if (x_click === undefined) {
        figure["layout"]["shapes"][0]["visible"] = false;
        figure["layout"]["shapes"][1]["visible"] = false;
      } else {
        figure["layout"]["shapes"][0]["visible"] = line_switch;
        figure["layout"]["shapes"][1]["visible"] = line_switch;
      }
      if (triggered_id === "interval-graph-update.n_intervals") {
        figure["data"][0]["x"] = x;
        figure["data"][0]["y"] = y;
        figure["data"][0]["z"] = z;
      }
      if (triggered_id === "heatmap.clickData") {
        figure["data"][1]["x"] = yz_scatter;
        figure["data"][1]["y"] = y_scatter;

        figure["data"][2]["x"] = x_scatter;
        figure["data"][2]["y"] = xz_scatter;

        figure["layout"]["shapes"][0]["x0"] = x_click;
        figure["layout"]["shapes"][0]["x1"] = x_click;

        figure["layout"]["shapes"][1]["y0"] = y_click;
        figure["layout"]["shapes"][1]["y1"] = y_click;
      }
      if (triggered_id === "modal_close.n_clicks") {
        figure["layout"]["xaxis"]["title"]["text"] = x_title;
        figure["layout"]["yaxis"]["title"]["text"] = y_title;
      }
      return figure;
    },
  },
});
