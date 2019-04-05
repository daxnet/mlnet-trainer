FROM mcr.microsoft.com/dotnet/core/aspnet:2.2-stretch-slim AS base
WORKDIR /app
EXPOSE 80

FROM mcr.microsoft.com/dotnet/core/sdk:2.2-stretch AS build
WORKDIR /src
COPY [".", "mlnet/"]
WORKDIR "/src/mlnet/mlnet_webapi"
RUN dotnet restore "mlnet_webapi.csproj"
RUN dotnet build "mlnet_webapi.csproj" -c Release -o /app

FROM build AS publish
RUN dotnet publish "mlnet_webapi.csproj" -c Release -o /app

FROM base AS final
WORKDIR /app
COPY --from=publish /app .
ENV ASPNETCORE_URLS="http://*:80"
CMD ["dotnet", "mlnet_webapi.dll"]
