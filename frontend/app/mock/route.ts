import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const body = await request.json();

  function addXHours(date: Date, hours: number) {
    return new Date(date.getTime() + hours * 60 * 60 * 1000);
  }

  function removeXHours(date: Date, hours: number) {
    return new Date(date.getTime() - hours * 60 * 60 * 1000);
  }

  function getPredictionTime(time: string) {
    const current = addXHours(new Date(), 1);
    switch (time) {
      case "CURRENT":
        return current;
      case "1HR":
        return removeXHours(current, 1);
      case "2HR":
        return removeXHours(current, 2);
      case "3HR":
        return removeXHours(current, 3);
      case "4HR":
        return removeXHours(current, 4);
      case "5HR":
        return removeXHours(current, 5);
      case "6HR":
        return removeXHours(current, 6);
      case "7HR":
        return removeXHours(current, 7);
      case "8HR":
        return removeXHours(current, 8);
      case "9HR":
        return removeXHours(current, 9);
      default:
        return current;
    }
  }

  return NextResponse.json({
    coin: body.coin || "BTC-USD",
    time: body.time || "1HR",
    model: body.model || "BTC-USD-CNN-1HR",
    prediction: { amount: 420.69, time: getPredictionTime(body.time) },
  });
}
