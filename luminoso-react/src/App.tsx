import React, {useState} from "react";
import MainPage from "./pages/mainPage";
import ErrorPage from "./pages/errorPage";
import HouseInfo from "./pages/houseInfo";

function App() {
    const [page, setPage] = useState(0);

    let RoutedPage;

    switch (page) {
        case 0:
            RoutedPage = <MainPage redirect={setPage}/>;
            break;
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
            RoutedPage = <HouseInfo house={page-1} redirect={setPage}/>;
            break;

        default:
            RoutedPage = <ErrorPage />;
            break;
    }

    return (
        <React.StrictMode>{RoutedPage}</React.StrictMode>
    )
}

export default App;