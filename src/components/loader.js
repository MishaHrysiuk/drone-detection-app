import React from "react";
import CircularProgress from "@mui/material/CircularProgress";

const Loader = (props) => {
    return (
        <div className="wrapper" {...props}>
            <CircularProgress />
            <p style={{ marginLeft: 10 }}>{props.children}</p>
        </div>
    );
};

export default Loader;
